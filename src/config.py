import os
import yaml
from pathlib import Path

current_path = Path(__file__).resolve()
project_root = current_path.parent.parent

# Load YAML Config
config_path = project_root / "config.yaml"

if config_path.exists():
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
else:
    raise FileNotFoundError(f"Config file not found at {config_path}")

DATA_DIR = project_root / cfg['paths']['data_raw']
TEST_DATA = project_root / cfg['paths']['data_processed']
MODEL_DIR = project_root / cfg['paths']['models_dir']
MODEL_DIR.mkdir(exist_ok=True)

# File Paths for Models
T_MODEL_PATH = MODEL_DIR / "t_learner_baseline.joblib"
C_MODEL_PATH = MODEL_DIR / "control_model.joblib"

# Feature Definition
FEATURES = ['user_freq'] + [f'f{i}' for i in range(20)]

# Hyperparameters 
PARAMS = cfg['hyperparameters']
SIM_PARAMS = cfg['simulation']

print(f"Configuration loaded from {config_path}")