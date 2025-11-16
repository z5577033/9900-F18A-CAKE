import logging
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

import pyarrow as pa
import pyarrow.feather as feather

# Quiet rpy2 warnings
os.environ.setdefault("R_ENABLE_JIT", "0")
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
rpy2_logger.setLevel(logging.ERROR)

# rpy2 on some images expects DataFrame.iteritems to exist
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

log = logging.getLogger(__name__)

def _to_df(X) -> pd.DataFrame:
    """
    Ensure X is a pandas DataFrame with string column names and
    a simple RangeIndex if needed.
    """
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(X)
    df.columns = df.columns.astype(str)
    # If index is not meaningful, keep as-is; R side reads sampleId from index we pass explicitly
    return df

def _to_series(y) -> pd.Series:
    """
    Ensure y is a pandas Series of dtype 'string' (labels).
    """
    if isinstance(y, pd.Series):
        s = y.copy()
    else:
        s = pd.Series(y)
    return s.astype("string")

class DifferentialMethylation(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer:
      fit(X: DataFrame/array, y: Series/array-of-labels) -> select probes via R/limma
      transform(X) -> X with selected probes only
    """
    def __init__(self, temp_dir="/tmp", top_n=50):
        self.temp_dir = temp_dir
        self.top_n = int(top_n)
        self.selected_probes_: list[str] | None = None

    def fit(self, X, y):
        X = _to_df(X)
        y = _to_series(y)

        # Build a per-class one-vs-rest differential methylation and union probes
        classes = y.unique().tolist()
        combined: set[str] = set()
        for idx, cls in enumerate(classes, start=1):
            print(f"probe identification for: {cls}, {idx} of {len(classes)} cancer types")
            condition = pd.Series(np.where(y == cls, cls, "otherCancerType"), dtype="string")
            probes = self._run_dm_once(X, condition)
            combined.update(probes)

        self.selected_probes_ = sorted(combined)
        print(f"Number of probes identified from differential methylation analyses: {len(self.selected_probes_)}")
        return self

    def transform(self, X):
        if self.selected_probes_ is None:
            raise NotFittedError("DifferentialMethylation must be fitted before transform().")
        X = _to_df(X)
        # Keep intersection in case some probes aren’t present in this split
        keep = [c for c in self.selected_probes_ if c in X.columns]
        print("transforming dataset to contain only differentially methylated features")
        if not keep:
            # Fallback: return original X to avoid empty-feature crash
            log.warning("DM produced no overlapping probes with X; returning original X.")
            return X
        return X[keep]

    # ---- internals ----

    def _run_dm_once(self, X: pd.DataFrame, condition: pd.Series) -> list[str]:
        """
        Prepare feather files, call R/limma via rpy2, return a list of probe names.
        """
        # Numeric cleanup for limma
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(axis="columns")
        # design frame: sampleId + cancerType (strings)
        design = pd.DataFrame(
            {"sampleId": pd.Series(X.index, dtype="string"),
             "cancerType": condition.values.astype("string")}
        )

        data_fn   = self._to_feather(X)
        design_fn = self._to_feather(design)

        try:
            probes = self._call_r_runDM(data_fn, design_fn)
        finally:
            with contextlib.suppress(Exception):
                os.remove(data_fn)
            with contextlib.suppress(Exception):
                os.remove(design_fn)

        # Limit to top_n if R returned more (defensive)
        if self.top_n and len(probes) > self.top_n:
            probes = probes[: self.top_n]
        return [str(p) for p in probes]

    def _to_feather(self, df: pd.DataFrame) -> str:
        schema_fields = []
        for col in df.columns:
            if col in ("sampleId", "cancerType", "Name"):
                pa_type = pa.string()
            else:
                pa_type = pa.float64()
            schema_fields.append((str(col), pa_type))
        table = pa.Table.from_pandas(df, schema=pa.schema(schema_fields), preserve_index=False)
        out = os.path.join(self.temp_dir, f"{uuid.uuid4().hex}.feather")
        feather.write_feather(table, out)
        return out

    def _call_r_runDM(self, data_filename: str, design_filename: str) -> list[str]:
        # Source the sibling R script
        r_script_path = Path(__file__).with_name("differentialMethylation.R")
        if not r_script_path.exists():
            raise FileNotFoundError(f"R script not found: {r_script_path}")

        robjects.r("options(rpy2_quiet = TRUE)")
        robjects.r["source"](str(r_script_path))
        runDM = robjects.globalenv["runDM"]

        with (robjects.default_converter + pandas2ri.converter).context():
            r_res = runDM(data_filename, design_filename)
            py_obj = robjects.conversion.get_conversion().rpy2py(r_res)

        # Accept DataFrame/Series/list/ndarray; extract probe IDs as strings
        if isinstance(py_obj, pd.DataFrame):
            if py_obj.index.size > 0:
                return list(py_obj.index.astype(str))
            # or fallback to any non-metadata columns
            return [c for c in map(str, py_obj.columns) if c not in ("sampleId", "cancerType", "Name")]
        if isinstance(py_obj, pd.Series):
            return [str(x) for x in py_obj.tolist()]
        if isinstance(py_obj, (list, tuple, np.ndarray)):
            return [str(x) for x in (py_obj.tolist() if hasattr(py_obj, "tolist") else list(py_obj))]
        log.warning("runDM returned unsupported type (%s); returning empty list", type(py_obj))
        return []
