"""Test configuration settings"""
from pathlib import Path

SQL_TEST_CONFIG = {
    "url": "postgresql://test:test@localhost:5432/test_db",
    "echo": False
}

TYPEDB_TEST_CONFIG = {
    "url": "localhost:1729",
    "database": "test_db"
} 


TEST_RESULTS_DIR = Path("/tmp/mch_test_results")


TEST_MODEL_CONFIGS = {
    "RandomForest": {
        "parameter_grid": {
            "modelGeneration__n_estimators": [2], 
            "modelGeneration__max_depth": [2],
            "modelGeneration__min_samples_split": [2],
            # "modelGeneration__class_weight": ["balanced"] 
        }
    }
}

TEST_PARAMETER_GRID = {
    model_type: model_cfg.get('parameter_grid', {})
    for model_type, model_cfg in TEST_MODEL_CONFIGS.items()
}