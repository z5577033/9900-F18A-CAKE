import os
import warnings
from pathlib import Path
import pandas as pd
import joblib
import json
import polars as pl

from mch.core.disease_tree import DiseaseTree
from mch.config.base_config import WORKING_DIRECTORY as PROJECT_ROOT
from mch.config.base_config import FREEZE as DEFAULT_FREEZE  # ← 引入默认 FREEZE
PROJECT_ROOT = Path(PROJECT_ROOT)
if str(PROJECT_ROOT).startswith("/dbfs/Workspace"):
    PROJECT_ROOT = Path(str(PROJECT_ROOT).replace("/dbfs", ""))
HERE_ROOT = Path(__file__).resolve().parents[3]
if "working_branch" in str(PROJECT_ROOT):
    PROJECT_ROOT = HERE_ROOT
TYPEDB_URI = "localhost:1729"
TYPEDB_DATABASE = "your_database_name"

# （src/mch/config → src/mch → src → working_branch）
DATA_ROOT = Path(os.environ.get(
    "MCH_DATA_DIR",
    "/Volumes/cb_prod/comp9300-9900-f18a-cake/9900-f18a-cake/data"
))
FREEZE = os.environ.get("MCH_FREEZE", DEFAULT_FREEZE)   # ← 修改：使用 DEFAULT_FREEZE 作为兜底
FREEZE_DIR = DATA_ROOT / FREEZE

def safe_file_exists(file_path):
    path = Path(file_path)
    if path.exists():
        return path
    warnings.warn(f"File not found: {file_path}")
    return None

def _maybe_copy_workspace_file(path: Path) -> Path:
    s = str(path)
    if s.startswith("/Workspace/"):
        try:
            local_tmp = Path("/databricks/driver") / path.name
            dbutils.fs.cp(f"file:{s}", f"file:{local_tmp}", True)
            return local_tmp
        except Exception:
            return path
    return path

def safe_load_csv(file_path, default=None):
    path = safe_file_exists(file_path)
    if not path:
        return default
    try:
        path = _maybe_copy_workspace_file(path)
        return pl.read_csv(path)
    except Exception as e:
        warnings.warn(f"Error loading CSV {file_path}: {e}")
        return default

def safe_load_joblib(file_path, default=None):
    path = safe_file_exists(file_path)
    if not path:
        return default
    try:
        path = _maybe_copy_workspace_file(path)
        with open(path, "rb") as f:
            return joblib.load(f)
    except Exception as e:
        warnings.warn(f"Error loading joblib {file_path}: {e}")
        return default

def safe_load_json(file_path, default=None):
    path = safe_file_exists(file_path)
    if not path:
        return default
    try:
        path = _maybe_copy_workspace_file(path)
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Error loading JSON {file_path}: {e}")
        return default

def load_data():
    """Load and cache all necessary data files with Spark + Polars compatibility."""
    from pyspark.sql import SparkSession
    import polars as pl

    spark = SparkSession.builder.getOrCreate()

    #  Unity Catalog 
    #  spark_df = spark.table(r"cb_prod.comp9300-9900-f18a-cake.filter_meth_mvalues_masked_subset_leukaemia.csv")

    # Polars DataFrame
    # pandas_df = pl.read_csv(
    #     "/Volumes/cb_prod/comp9300-9900-f18a-cake/9900-f18a-cake/data/mvalue_outputs_masked_subset_leukaemia_subsampled/MValue_polaris_pivot_0.csv"
    # )

    mvalue_df = pl.read_csv(
            "/Volumes/cb_prod/comp9300-9900-f18a-cake/9900-f18a-cake/data/mvalue_outputs_masked/MValue_concat.csv"
    )

    # tree/joblib & colorProfiles.json  —— 固定项目目录
    tree_path  = PROJECT_ROOT / "data" / "freeze0525" / "diseaseTree_mapped.joblib"
    color_path = PROJECT_ROOT / "data" / "colorProfiles.json"
    main_tree = safe_load_joblib(tree_path)
    color_profiles = safe_load_json(color_path)

    return mvalue_df, main_tree, color_profiles, main_tree

mvalue_df, main_tree, color_profiles, disease_tree = load_data()

model_directory = f"{FREEZE_DIR}/models/"
model_parameter_directory = f"{FREEZE_DIR}/model_parameters/"
full_model_directory = f"{FREEZE_DIR}/full_models/"
tree_directory = f"{FREEZE_DIR}/trees/"
embedding_directory = f"{FREEZE_DIR}/embeddings/"
who_book_file = f"{FREEZE_DIR}/who_book.json"
cancer_type_file = f"{FREEZE_DIR}/cancer_types.csv"

base_mvalue_df = mvalue_df

def validate_critical_data():
    critical_missing = []
    if mvalue_df is None:
        critical_missing.append("mvalue_df")
    if main_tree is None:
        critical_missing.append("main_tree")
    if color_profiles is None:
        critical_missing.append("color_profiles")
    if base_mvalue_df is None:
        critical_missing.append("base_mvalue_df")

    if critical_missing:
        raise RuntimeError(
            "Critical data missing: " + ", ".join(critical_missing) +
            f"\nDATA_ROOT={DATA_ROOT}\nFREEZE={FREEZE}"
        )
    return True

data_validation_passed = validate_critical_data()

print("[OK] All critical data loaded successfully.")
print(f"[INFO] mvalue_df shape: {mvalue_df.shape}")
print(f"[INFO] base_mvalue_df shape: {base_mvalue_df.shape}")
print(f"[INFO] Data root: {DATA_ROOT}")
