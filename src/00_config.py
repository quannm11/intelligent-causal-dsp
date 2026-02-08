import os
from pathlib import Path

# Project Files
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "v2"
RESULT_DIR = PROJECT_ROOT / "results"

# Standardized Features 
BASE_FEATURES = [f'f{i}' for i in range(12)]
ENGINEERED_FEATURES = ['user_freq', 'f3_sq', 'f8_sq', 'f6_sq', 'f3_f6_inter', 'f2_f9_inter']
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

# File Path Definitions
TRAIN_DATA = DATA_DIR / "v2_engineered" / "train_data"
VAL_DATA = DATA_DIR / "v2_engineered" / "val_data"
TEST_DATA = DATA_DIR / "v2_engineered" / "test_data"

T_MODEL_PATH = MODEL_DIR / "t_learner_treatment.joblib"
C_MODEL_PATH = MODEL_DIR / "t_learner_control.joblib"

CONVERSION_VALUE = 100.0
SEED = 42

for path in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    path.mkdir(parents=True, exist_ok=True)