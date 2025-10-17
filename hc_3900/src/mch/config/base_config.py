import yaml
from pathlib import Path

cfg_path = Path(__file__).with_suffix(".yaml")
config = yaml.safe_load(cfg_path.read_text())

FREEZE = config.get("freeze")
FREEZE_NUMBER = config.get("freeze_number")
WORKING_DIRECTORY = config.get("working_directory")
credential_file = config.get("credential_file")

MVALUES_PATH = config.get("mvalues_path")
ID_COLUMN = config.get("id_column", "biosample_id")
