from pathlib import Path
import re
import sys
import os
import joblib

# 尝试 polars，失败则使用 pandas
try:
    import polars as pl
except Exception:
    pl = None

try:
    import pandas as pd
except Exception:
    pd = None

PROJECT_SRC = "/Workspace/9900-f18a-cake/working_branch/src"
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

import joblib

JOBLIB_PATH = "/Workspace/9900-f18a-cake/working_branch/data/freeze0525/diseaseTree_mapped.joblib"
CSV_IN_PATH = "/Volumes/cb_prod/comp9300-9900-f18a-cake/9900-f18a-cake/data/ mvalue_outputs_masked_subset_leukaemia/MValue_concat.csv"
CSV_OUT_PATH = "/Volumes/cb_prod/comp9300-9900-f18a-cake/9900-f18a-cake/data/mvalue_outputs_masked_subset_leukaemia/MValue_concat_fake.csv"

ID_CANDIDATES = ["biosample_id", "sample_id", "id", "biosample", "sample", "case_id"]
ID_PATTERN = re.compile(r".*_T_.*_M$")  # 2J0D2U4J_T_XJAFP6HU_M
DEFAULT_ID_COL = "biosample_id"

def log(msg: str):
    print(f"[fakeDatagenerate] {msg}")

def is_candidate_id(x) -> bool:
    try:
        s = str(x).strip()
    except Exception:
        return False
    if not s:
        return False
    return bool(ID_PATTERN.match(s))

def deep_scan_ids(obj, ids: set, _depth=0, _maxdepth=8):
    """深度扫描嵌套对象，提取匹配 *_T_*_M 的字符串为样本 ID。"""
    if obj is None or _depth > _maxdepth:
        return
    if isinstance(obj, str):
        if is_candidate_id(obj):
            ids.add(obj.strip())
        return
    if isinstance(obj, (list, tuple, set)):
        for it in obj:
            deep_scan_ids(it, ids, _depth + 1, _maxdepth)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and is_candidate_id(k):
                ids.add(k.strip())
            deep_scan_ids(v, ids, _depth + 1, _maxdepth)
        return
    try:
        import numpy as np  # noqa
        if hasattr(obj, "dtype") or "numpy" in type(obj).__module__:
            try:
                it = obj.tolist()
            except Exception:
                it = None
            if it is not None:
                deep_scan_ids(it, ids, _depth + 1, _maxdepth)
                return
    except Exception:
        pass

    if hasattr(obj, "__dict__"):
        try:
            deep_scan_ids(vars(obj), ids, _depth + 1, _maxdepth)
        except Exception:
            pass

def load_sample_ids_from_joblib(joblib_path: str) -> list[str]:
    log(f"Loading joblib: {joblib_path}")
    obj = joblib.load(joblib_path)
    ids = set()
    deep_scan_ids(obj, ids)
    ids_list = sorted(ids)
    log(f"Found {len(ids_list)} sample IDs from joblib (pattern *_T_*_M).")
    return ids_list

def read_csv_any(csv_path: str):
    """返回 (df, engine) ，engine in {'polars','pandas'}。"""
    if pl is not None:
        log("Reading CSV with polars...")
        df = pl.read_csv(csv_path, ignore_errors=True)
        return df, "polars"
    if pd is not None:
        log("Reading CSV with pandas...")
        df = pd.read_csv(csv_path)
        return df, "pandas"
    raise RuntimeError("Neither polars nor pandas is available in this environment.")

def ensure_id_column(df, engine: str) -> str:
    """找到或创建样本 ID 列，返回列名。"""
    def cols_of(obj):
        return obj.columns if engine == "polars" else list(obj.columns)

    cols = [c.strip() if isinstance(c, str) else c for c in cols_of(df)]
    for cand in ID_CANDIDATES:
        for c in cols:
            if isinstance(c, str) and c.lower() == cand.lower():
                log(f"Using existing ID column: {c}")
                return c


    id_col = DEFAULT_ID_COL
    log(f"No known ID column found. Creating column: {id_col}")
    if engine == "polars":
        if id_col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(id_col))
        return id_col, df  
    else:
        if id_col not in df.columns:
            df[id_col] = None
        return id_col, df

def overwrite_ids(df, engine: str, id_col: str, new_ids: list[str]):
    n_rows = df.height if engine == "polars" else len(df)
    n_ids = len(new_ids)
    n = min(n_rows, n_ids)
    log(f"Overwriting first {n} rows in '{id_col}' with joblib IDs.")
    if n == 0:
        return df

    if engine == "polars":
        repl = pl.Series(name=id_col, values=new_ids[:n])
        left = df.slice(0, n).with_columns(repl)
        right = df.slice(n, n_rows - n)
        df_out = pl.concat([left, right], how="vertical_relaxed")
        return df_out
    else:
        df.loc[: n - 1, id_col] = new_ids[:n]
        return df

def write_csv_any(df, engine: str, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing CSV -> {out_path}")
    if engine == "polars":
        df.write_csv(out_path)
    else:
        df.to_csv(out_path, index=False)
    log("Done.")

def main():
    # 1) 取样本 ID（深度扫描）
    sample_ids = load_sample_ids_from_joblib(JOBLIB_PATH)

    # 2) 读 CSV
    df, engine = read_csv_any(CSV_IN_PATH)

    # 3) 找/建 ID 列
    if engine == "polars":
        # ensure_id_column 里若创建列，需要接收新的 df
        id_col = None
        res = ensure_id_column(df, engine)
        if isinstance(res, tuple):
            id_col, df = res
        else:
            id_col = res
    else:
        id_col = ensure_id_column(df, engine)
        if isinstance(id_col, tuple):
            # pandas 分支上不应出现，但稳妥处理
            id_col, df = id_col

    # 4) overwrite 模式：覆盖前 N 行 ID
    df = overwrite_ids(df, engine, id_col, sample_ids)

    # 5) 写出
    write_csv_any(df, engine, CSV_OUT_PATH)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        # 在 Databricks 中方便快速定位报错
        import traceback
        traceback.print_exc()
        sys.exit(1)