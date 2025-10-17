"""
mch.config.settings
-------------------
Minimal runtime settings/helpers for inference:

- Reads core config values from base_config.yaml via base_config.py:
    freeze, freeze_number, working_directory, mvalues_path, id_column
- Loads a trained sklearn model from: {freeze}/rf_baseline.joblib
- (Optional) Loads a feature list from: {freeze}/feature_list.txt
- Provides helpers:
    - read_one_sample(sample_id): return a 1-row DataFrame from the big table
    - vectorize_for_model(df_row): align/impute -> numpy array (X, cols)
    - get_model_classes(): list of class labels if the model is loaded

This keeps memory small by reading exactly one row on demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import warnings

import joblib
import pandas as pd

# ── Config values ──────────────────────────────────────────────────────────────
from mch.config.base_config import (
    WORKING_DIRECTORY,
    FREEZE,
    FREEZE_NUMBER,
    MVALUES_PATH,
    ID_COLUMN,
)

# ── Paths & small utilities ────────────────────────────────────────────────────

def _p(*parts) -> Path:
    """Resolve a path under WORKING_DIRECTORY."""
    return Path(WORKING_DIRECTORY).joinpath(*parts)

FREEZE_DIR = _p(FREEZE) if FREEZE else Path(".")
MODEL_PATH = FREEZE_DIR / "rf_baseline.joblib"          # adjust if your model name differs
FEATURE_LIST_PATH = FREEZE_DIR / "feature_list.txt"     # optional; one probe/column per line

def safe_load_joblib(path: Path, default=None):
    """Load a joblib artifact safely, returning default on failure."""
    try:
        if path and path.exists():
            return joblib.load(path)
    except Exception as e:
        warnings.warn(f"joblib load failed for {path}: {e}")
    return default

def load_feature_names_from_file(path: Path) -> Optional[List[str]]:
    """Return a list of feature/column names if file exists; else None."""
    if path and path.exists():
        return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return None

# ── Global, lazily-used artifacts ──────────────────────────────────────────────

MODEL = safe_load_joblib(MODEL_PATH, default=None)
FEATURE_NAMES = load_feature_names_from_file(FEATURE_LIST_PATH)

# ── Data access helpers ────────────────────────────────────────────────────────

def _read_one_parquet_row(parquet_path: Path, sample_id: str, use_feature_list: bool) -> pd.DataFrame:
    """
    Read a single row from a Parquet dataset, filtering on ID_COLUMN.
    Requires pyarrow to be installed.
    """
    try:
        import pyarrow.dataset as ds  # type: ignore
        dataset = ds.dataset(parquet_path)
        columns = [ID_COLUMN] + (FEATURE_NAMES if (use_feature_list and FEATURE_NAMES) else [])
        table = dataset.to_table(filter=ds.field(ID_COLUMN) == sample_id,
                                 columns=columns if use_feature_list and FEATURE_NAMES else None)
        df = table.to_pandas()
        if df.empty and (use_feature_list and FEATURE_NAMES):
            # Fallback: fetch all columns for this row
            table = dataset.to_table(filter=ds.field(ID_COLUMN) == sample_id)
            df = table.to_pandas()
        return df
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Parquet read requires 'pyarrow'. Install it or convert your M-values to CSV, "
            "or set mvalues_path to a CSV file."
        ) from e

def _read_one_csv_row(csv_path: Path, sample_id: str, use_feature_list: bool) -> pd.DataFrame:
    """
    Stream a large CSV in chunks to find one sample row.
    If a feature list is provided, attempt to read only those columns plus ID.
    """
    # If we specify usecols for chunked reading, pandas can prune at parse time.
    usecols = None
    if use_feature_list and FEATURE_NAMES:
        # Always include ID_COLUMN
        usecols = [ID_COLUMN] + list(FEATURE_NAMES)

    for chunk in pd.read_csv(csv_path, chunksize=50_000, usecols=usecols):
        hit = chunk.loc[chunk[ID_COLUMN] == sample_id]
        if not hit.empty:
            return hit

    # not found
    return pd.DataFrame(columns=usecols if usecols else [])

def read_one_sample(sample_id: str, use_feature_list: bool = True) -> pd.DataFrame:
    """
    Return a single-row DataFrame for the given sample_id.
    Prefers loading only columns present in FEATURE_NAMES (if provided),
    otherwise falls back to all columns (minus ID at vectorization time).
    """
    if not MVALUES_PATH:
        raise RuntimeError("mvalues_path is not set in base_config.yaml")

    data_path = _p(MVALUES_PATH)

    if not data_path.exists():
        raise FileNotFoundError(f"mvalues_path not found: {data_path}")

    if str(data_path).lower().endswith(".parquet"):
        df = _read_one_parquet_row(data_path, sample_id, use_feature_list)
    else:
        df = _read_one_csv_row(data_path, sample_id, use_feature_list)

    if df.empty:
        raise KeyError(f"Sample '{sample_id}' not found in {data_path}")

    return df

def vectorize_for_model(df_row: pd.DataFrame) -> Tuple["np.ndarray", List[str]]:
    """
    Convert a 1-row DataFrame to a numeric array X (shape: 1 x n_features),
    aligned to training-time columns.

    If FEATURE_NAMES is present, we enforce that order.
    Else, we use all columns except the ID column, in file order.

    NaNs are mean-imputed per feature.
    """
    import numpy as np  # local import keeps module import light
    if FEATURE_NAMES:
        cols = [c for c in FEATURE_NAMES if c in df_row.columns]
        # If some expected features are missing from the row, warn once.
        missing = [c for c in FEATURE_NAMES if c not in df_row.columns]
        if missing:
            warnings.warn(f"{len(missing)} expected feature(s) missing in row; proceeding with available columns.")
    else:
        cols = [c for c in df_row.columns if c != ID_COLUMN]

    if not cols:
        raise RuntimeError("No feature columns available after alignment. "
                           "Ensure your feature_list.txt matches training or remove it to use all columns.")

    X = df_row[cols].to_numpy(dtype=float, copy=False)
    # mean-impute per feature
    col_means = np.nanmean(X, axis=0, keepdims=True)
    X = np.nan_to_num(X, nan=col_means)

    return X, cols

def get_model_classes() -> Optional[List[str]]:
    """Return class labels if the model exposes them."""
    if MODEL is not None and hasattr(MODEL, "classes_"):
        try:
            return MODEL.classes_.tolist()
        except Exception:
            return list(MODEL.classes_)
    return None

# ── Convenience for health checks ─────────────────────────────────────────────

def health_summary() -> dict:
    return {
        "working_directory": str(WORKING_DIRECTORY),
        "freeze_dir": str(FREEZE_DIR),
        "freeze_number": FREEZE_NUMBER,
        "model_path": str(MODEL_PATH),
        "model_loaded": MODEL is not None,
        "feature_list": str(FEATURE_LIST_PATH) if FEATURE_LIST_PATH.exists() else None,
        "mvalues_path": str(_p(MVALUES_PATH)) if MVALUES_PATH else None,
        "id_column": ID_COLUMN,
        "classes": get_model_classes(),
    }
