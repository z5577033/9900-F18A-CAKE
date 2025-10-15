import yaml
from pathlib import Path

# Load variables from modelTrainingConfig.yaml
def load_model_config():
    """Load model training configuration from YAML file."""
    yaml_file_path = Path(__file__).resolve().parent / 'modelTrainingConfig.yaml'
    
    try:
        with open(yaml_file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
    except FileNotFoundError:
        raise Exception(f"Configuration file not found: {yaml_file_path}")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")
    
    return config

def load_refinement_config():
    """Legacy function name - redirects to load_model_config for backward compatibility."""
    return load_model_config()

# Load the configuration at module import time
try:
    refinement_config = load_model_config()
    
    # Expose variables for easy access
    resultsDirectory = refinement_config.get('resultsDirectory')
    parameter_grid = refinement_config.get('parameter_grid')
    grid_search_config = refinement_config.get('grid_search_config')
    default_model_type = refinement_config.get('default_model_type')
    
except Exception as e:
    print(f"Warning: Could not load model configuration: {e}")
    # Provide fallback values
    resultsDirectory = "results/models"
    parameter_grid = {}
    grid_search_config = {
        'n_splits': 5,
        'scoring': 'accuracy',
        'verbose': 1,
        'n_jobs': -1,
        'error_score': 'raise'
    }
    default_model_type = "svm"


#filteredMValueFile: /media/storage/bcurran/classifiers/methylation/data/methylationMValuesFiltered.parquet
#datafile = "/media/storage/bcurran/classifiers/methylation/data/methylationHG38FilteredMValue.h5ad"
#filteredDataFile: /data/projects/classifiers/methylation/data/baseMethylationBetaValuesFiltered.parquet


