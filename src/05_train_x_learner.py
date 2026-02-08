# src/05_train_x_learner.py
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import logging
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import TRAIN_DATA, VAL_DATA, MODEL_DIR, FEATURES, TARGET, TREATMENT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def train_x_learner():
    logger.info("--- TRAINING X-LEARNER ---")
    
    train_df = pd.read_parquet(TRAIN_DATA)
    
    # --- Base T-Learner ---
    logger.info("Training Base Models (Mu_0, Mu_1)")
    
    X_t = train_df[train_df[TREATMENT] == 1][FEATURES]
    y_t = train_df[train_df[TREATMENT] == 1][TARGET]
    
    X_c = train_df[train_df[TREATMENT] == 0][FEATURES]
    y_c = train_df[train_df[TREATMENT] == 0][TARGET]
    
    mu_1 = xgb.XGBRegressor(objective='binary:logistic', n_estimators=100, max_depth=4)
    mu_1.fit(X_t, y_t)
    
    mu_0 = xgb.XGBRegressor(objective='binary:logistic', n_estimators=100, max_depth=4)
    mu_0.fit(X_c, y_c)
    
    # --- Impute Counterfactuals ---
    logger.info("Imputing Treatment Effects")
    
    # For Treated units: D_1 = Y_observed - Mu_0(X)
    d_1 = y_t - mu_0.predict(X_t)
    
    # For Control units: D_0 = Mu_1(X) - Y_observed
    d_0 = mu_1.predict(X_c) - y_c
    
    # --- Train Second-Stage Models (Tau) ---
    logger.info("Training Second-Stage Effect Models (Tau_0, Tau_1)")
    
    tau_1 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4)
    tau_1.fit(X_t, d_1) # Learn effect from treated perspective
    
    tau_0 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4)
    tau_0.fit(X_c, d_0) # Learn effect from control perspective
    
    # --- Propensity Weighting (The Gating Network) ---
    logger.info("Learning Propensity Score (g)")
    g_model = LogisticRegression(solver='liblinear')
    g_model.fit(train_df[FEATURES], train_df[TREATMENT])
    
    save_path = MODEL_DIR / "x_learner_artifacts.joblib"
    artifacts = {
        "tau_1": tau_1,
        "tau_0": tau_0,
        "g_model": g_model
    }
    joblib.dump(artifacts, save_path)
    logger.info(f"X-Learner saved to {save_path}")

if __name__ == "__main__":
    train_x_learner()