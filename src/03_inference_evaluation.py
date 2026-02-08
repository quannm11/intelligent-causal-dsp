import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import TEST_DATA, RESULT_DIR, FEATURES, T_MODEL_PATH, C_MODEL_PATH, PREDICTIONS_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference():
    logger.info("Loading Test Data")
    try:
        test_df = pd.read_parquet(TEST_DATA)
        logger.info(f"Test data loaded: {test_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return

    logger.info("Loading Models...")
    try:
        model_t = joblib.load(T_MODEL_PATH)
        model_c = joblib.load(C_MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    logger.info("Calculating Uplift Scores (CATE)")
    X_test = test_df[FEATURES]
    
    prob_t = model_t.predict_proba(X_test)[:, 1]
    prob_c = model_c.predict_proba(X_test)[:, 1]
    
    test_df['prob_treatment'] = prob_t
    test_df['prob_control'] = prob_c
    test_df['uplift_score'] = prob_t - prob_c
    
    save_path = PREDICTIONS_PATH
    logger.info(f"Saving predictions to {save_path}")
    test_df.to_parquet(save_path)
    
    avg_uplift = test_df['uplift_score'].mean()
    positive_uplift_pct = (test_df['uplift_score'] > 0).mean() * 100
    
    print("\n" + "="*40)
    print(f"RESULTS SUMMARY")
    print(f"="*40)
    print(f"Average Predicted Uplift: {avg_uplift:.6f}")
    print(f"Persuadable Population:   {positive_uplift_pct:.2f}% (Customers worth bidding on)")
    print(f"="*40 + "\n")

if __name__ == "__main__":
    run_inference()