import yaml
from pathlib import Path

def load_refinement_config():
    yaml_file_path = Path(__file__).resolve().parent / 'model_training_config.yaml'
    try:
        with open(yaml_file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
    except FileNotFoundError:
        raise Exception(f"Configuration file not found: {yaml_file_path}")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")
    return config

refinement_config = load_refinement_config()

resultsDirectory = refinement_config.get('resultsDirectory')

model_configs = refinement_config.get('model_configs', {})
if 'random_forest' in model_configs:
    parameter_grid = model_configs['random_forest'].get('parameter_grid', {})
    print(f"[modelTrainingParameters] Using Random Forest parameters")
    print(f"[modelTrainingParameters] Parameters: {list(parameter_grid.keys())}")
else:
    for model_type, model_cfg in model_configs.items():
        param_grid = model_cfg.get('parameter_grid', {})
        if param_grid:
            parameter_grid = param_grid
            print(f"[modelTrainingParameters] Using {model_type} parameters")
            break