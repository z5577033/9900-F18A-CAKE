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

# (src/mch/config -> src/mch -> src -> working_branch)
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
    """Load data using Smart Sampling (Random IDs) to prevent OOM on 128GB node."""
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, rand
    import polars as pl
    import gc 

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    table_name = "cb_prod.`comp9300-9900-f18a-cake`.filter_meth_mvalues_masked_subset_leukaemia"
    print(f"Connecting to Unity Catalog table: {table_name}")
    

    full_spark_df = spark.table(table_name)

    # ==========================================================================
    #  Smart sampling logic 
    # ==========================================================================
    print("Fetching list of unique Sample IDs...")
    
    # 1. Query only unique biosample_id 
    unique_ids_df = full_spark_df.select("biosample_id").distinct()
    
    # 2. Shuffle IDs randomly
    SAMPLE_COUNT = 100
    print(f"Randomly selecting {SAMPLE_COUNT} distinct samples...")
    
    sampled_rows = unique_ids_df.orderBy(rand()).limit(SAMPLE_COUNT).collect()
    target_ids = [row['biosample_id'] for row in sampled_rows]
    
    print(f" Selected {len(target_ids)} samples. Sample IDs (first 5): {target_ids[:5]}")

    # 3. Fetch only the detailed data for these samples
    print("Fetching data rows ONLY for selected samples...")
    spark_df = full_spark_df.filter(col("biosample_id").isin(target_ids))

    # ==========================================================================

    # Conversion process
    try:
        print("Converting to Pandas (Safe Mode)...")
        pandas_df = spark_df.toPandas()
        
        print(f"Converting to Polars (Rows: {len(pandas_df)})...")
        mvalue_df = pl.from_pandas(pandas_df)
        
        del pandas_df
        gc.collect()
    except Exception as e:
        print(f" MEMORY ERROR: {e}")
        print("It is recommended to reduce the SAMPLE_COUNT above (e.g., change to 30).")
        raise e

    # PIVOT operation
    if "Name" in mvalue_df.columns and "MValue" in mvalue_df.columns:
        print(" Pivoting data from Long to Wide format...")
        try:
            mvalue_df = mvalue_df.pivot(
                index="biosample_id", 
                columns="Name", 
                values="MValue",
                aggregate_function=None
            )
            print(f" Pivot successful! Final Data Shape: {mvalue_df.shape}")
        except Exception as e:
            print(f" Pivot warning: {e}. Retrying with aggregation...")
            mvalue_df = mvalue_df.pivot(
                index="biosample_id", 
                columns="Name", 
                values="MValue",
                aggregate_function="mean"
            )
            print(f" Pivot successful! Final Data Shape: {mvalue_df.shape}")

    # Load Tree
    tree_path = ROOT / "data" / "freeze0525" / "diseaseTree_mapped.joblib"
    color_path = ROOT / "data" / "colorProfiles.json"
    main_tree = safe_load_joblib(tree_path)
    color_profiles = safe_load_json(color_path)

    return mvalue_df, main_tree, color_profiles, main_tree

# ==============================================================================
# Global call
# ==============================================================================
print("Initializing Global Data Load (Smart Sampling)...")
mvalue_df, main_tree, color_profiles, disease_tree = load_data()
print("Global Data Load Complete.")

# ==============================================================================
# Global call
# ==============================================================================
print("Initializing Global Data Load (FULL DATASET)...")
mvalue_df, main_tree, color_profiles, disease_tree = load_data()
print("Global Data Load Complete.")

# ==============================================================================
# [CRITICAL FIX] This line executes the loading and assigns the global variables
# ==============================================================================
print("Initializing Global Data Load...")
mvalue_df, main_tree, color_profiles, disease_tree = load_data()
print("Global Data Load Complete.")
# ==============================================================================

model_directory = f"{FREEZE_DIR}/models/"
model_parameter_directory = f"{FREEZE_DIR}/model_parameters/"
full_model_directory = f"{FREEZE_DIR}/full_models/"
tree_directory = f"{FREEZE_DIR}/trees/"
embedding_directory = f"{FREEZE_DIR}/embeddings/"
who_book_file = f"{FREEZE_DIR}/who_book.json"
cancer_type_file = f"{FREEZE_DIR}/cancer_types.csv"

# base_mvalue_df = safe_load_csv(f"{FREEZE_DIR}/MValue_concat_1.csv")
base_mvalue_df = None 

def validate_critical_data():
    critical_missing = []
    # Now mvalue_df is properly defined above, so this check will work
    if mvalue_df is None: critical_missing.append("mvalue_df")
    if main_tree is None: critical_missing.append("main_tree")
    if color_profiles is None: critical_missing.append("color_profiles")
    
    if critical_missing:
        warnings.warn(f"Critical data missing: {', '.join(critical_missing)}")
        return False
    return True

data_validation_passed = validate_critical_data()