import yaml
from pathlib import Path

# 直接定义项目根目录下的正确配置文件路径
# 无论这个脚本在哪里被调用，它都会去正确的地方寻找
config_path = Path("/app/config/base_config.yaml")

# 确保文件存在，如果不存在就给出清晰的错误提示
if not config_path.is_file():
    raise FileNotFoundError(
        f"配置文件未找到！请确保 '{config_path}' 文件存在。"
    )

with open(config_path) as f:
    config = yaml.safe_load(f)

FREEZE = config.get("freeze")
FREEZE_NUMBER = config.get("freeze_number")
WORKING_DIRECTORY = config.get("working_directory")
credential_file = config.get("credential_file")