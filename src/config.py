import os
from pathlib import Path
from dotenv import load_dotenv

# Project Root 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Directory Definitions
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models" / "v2"))
RESULT_DIR = Path(os.getenv("RESULT_DIR", PROJECT_ROOT / "results"))

# Ensure directories exist
for path in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# File Path Definitions
RAW_DATA_PATH = DATA_DIR / "criteo_uplift.csv.gz"
TRAIN_DATA = DATA_DIR / "v2_engineered" / "train_data.parquet"
VAL_DATA = DATA_DIR / "v2_engineered" / "val_data.parquet"
TEST_DATA = DATA_DIR / "v2_engineered" / "test_data.parquet"
PREDICTIONS_PATH = RESULT_DIR / "final_test_predictions.parquet"

# Model Paths
T_MODEL_PATH = MODEL_DIR / "t_learner_treatment.joblib"
C_MODEL_PATH = MODEL_DIR / "t_learner_control.joblib"

# Feature Definitions 
BASE_FEATURES = [f'f{i}' for i in range(12)]
ENGINEERED_FEATURES = [
    'user_freq', 
    'f3_sq', 'f8_sq', 'f6_sq', 
    'f3_f6_inter', 'f2_f9_inter'
]
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
TARGET = 'conversion'
TREATMENT = 'treatment'

# Global Constants
CONVERSION_VALUE = float(os.getenv("CONVERSION_VALUE", 100.0))
SEED = 42