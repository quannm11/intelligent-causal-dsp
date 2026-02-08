# src/04_refutation_test.py
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys
from pathlib import Path

# Path Setup
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import TRAIN_DATA, FEATURES, TARGET, TREATMENT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_placebo_test():
    logger.info("--- STARTING PLACEBO TEST (REFUTATION) ---")
    
    df = pd.read_parquet(TRAIN_DATA).sample(frac=0.2, random_state=42) # Downsample for speed
    logger.info(f"Loaded {len(df)} rows. Shuffling Treatment...")
    
    df['placebo_treatment'] = np.random.permutation(df[TREATMENT].values)
    
    # Train T-Learner on Noise
    logger.info("Training T-Learner on Placebo Data")
    
    # Split by Placebo Treatment
    idx_t = df['placebo_treatment'] == 1
    idx_c = df['placebo_treatment'] == 0
    
    model_t = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss')
    model_t.fit(df.loc[idx_t, FEATURES], df.loc[idx_t, TARGET])
    
    model_c = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss')
    model_c.fit(df.loc[idx_c, FEATURES], df.loc[idx_c, TARGET])
    
    # Predict Uplift
    pred_t = model_t.predict_proba(df[FEATURES])[:, 1]
    pred_c = model_c.predict_proba(df[FEATURES])[:, 1]
    uplift = pred_t - pred_c
    
    avg_uplift = np.mean(uplift)
    logger.info(f"Average Placebo Uplift: {avg_uplift:.5f} (Should be close to 0)")
    
    # Check Feature Importance
    # If a feature is highly important here, it's leaking info.
    if abs(avg_uplift) > 0.01:
        logger.error("FAILED: Significant signal found in random noise. Check for leakage!")
    else:
        logger.info("PASSED: Placebo uplift is indistinguishable from noise.")

if __name__ == "__main__":
    run_placebo_test()