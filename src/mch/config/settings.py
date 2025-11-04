import warnings
from pathlib import Path
import pandas as pd
import joblib
import json
import polars as pl

from mch.core.disease_tree import DiseaseTree
from mch.config.base_config import FREEZE, FREEZE_NUMBER, WORKING_DIRECTORY

TYPEDB_URI = "localhost:1729"
TYPEDB_DATABASE = "your_database_name"

# （src/mch/config → src/mch → src → working_branch）
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
WORKING_DIRECTORY = ROOT
FREEZE_DIR = DATA_DIR / FREEZE

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
        "/Volumes/cb_prod/comp9300-9900-f18a-cake/9900-f18a-cake/data/Trainable_data/MValue_concat_overwrite.csv"
    )

    # tree/joblib & colorProfiles.json 
    tree_path = ROOT / "data" / "freeze0525" / "diseaseTree_mapped.joblib"
    color_path = ROOT / "data" / "colorProfiles.json"
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

base_mvalue_df = safe_load_csv(f"{FREEZE_DIR}/MValue_concat_1.csv")

def validate_critical_data():
    critical_missing = []
    if mvalue_df is None: critical_missing.append("mvalue_df")
    if main_tree is None: critical_missing.append("main_tree")
    if color_profiles is None: critical_missing.append("color_profiles")
    if base_mvalue_df is None: critical_missing.append("base_mvalue_df")
    if critical_missing:
        warnings.warn(f"Critical data missing: {', '.join(critical_missing)}")
        return False
    return True

data_validation_passed = validate_critical_data()
