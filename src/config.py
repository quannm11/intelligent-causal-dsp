import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Directory Definitions
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models" / "v2"))
RESULT_DIR = Path(os.getenv("RESULT_DIR", PROJECT_ROOT / "results"))

# File Path Definitions
TRAIN_DATA = DATA_DIR / "v2_engineered" / "train_data"
TEST_DATA = DATA_DIR / "v2_engineered" / "test_data"
PREDICTIONS_PATH = RESULT_DIR / "final_test_predictions.parquet"

# Model Paths
T_MODEL_PATH = MODEL_DIR / "t_learner_treatment.joblib"
C_MODEL_PATH = MODEL_DIR / "t_learner_control.joblib"

CONVERSION_VALUE = float(os.getenv("CONVERSION_VALUE", 100.0))

# Ensure directories exist
for path in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    path.mkdir(parents=True, exist_ok=True)