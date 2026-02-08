import os
import sys
import argparse
import logging
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import roc_auc_score
from pathlib import Path


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.config import TRAIN_DATA, VAL_DATA, T_MODEL_PATH, C_MODEL_PATH, FEATURES, TARGET, TREATMENT
except ImportError:
    from config import TRAIN_DATA, VAL_DATA, T_MODEL_PATH, C_MODEL_PATH, FEATURES, TARGET, TREATMENT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parses command line arguments for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Train T-Learner Uplift Models (XGBoost)")
    
    # Model Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Step size shrinkage used in update")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum depth of a tree")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of boosting rounds")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio of the training instances")
    
    return parser.parse_args()

def train_and_calibrate(args):
    logger.info("Loading Training and Validation data")
    
    try:
        # Load Parquet files 
        train_df = pd.read_parquet(TRAIN_DATA)
        val_df = pd.read_parquet(VAL_DATA)
        logger.info(f"Data Loaded. Train: {train_df.shape}, Val: {val_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    groups = [(1, "treatment", T_MODEL_PATH), (0, "control", C_MODEL_PATH)]
    
    for group_id, label, save_path in groups:
        logger.info(f"--- Processing {label.upper()} Model (Group={group_id}) ---")
        
        X_train = train_df[train_df[TREATMENT] == group_id][FEATURES]
        y_train = train_df[train_df[TREATMENT] == group_id][TARGET]
        
        X_val = val_df[val_df[TREATMENT] == group_id][FEATURES]
        y_val = val_df[val_df[TREATMENT] == group_id][TARGET]
        
        X_full = pd.concat([X_train, X_val], axis=0)
        y_full = pd.concat([y_train, y_val], axis=0)
        
        split_index = ([-1] * len(X_train)) + ([0] * len(X_val))
        pds = PredefinedSplit(test_fold=split_index)

        logger.info(f"Training XGBoost on {len(X_train)} samples with {len(FEATURES)} features")
        logger.debug(f"Params: LR={args.learning_rate}, Depth={args.max_depth}, Est={args.n_estimators}")

        base_model = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            tree_method='hist', 
            random_state=42
        )
                
        logger.info("Calibrating probabilities using Isotonic Regression")
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model, 
            method='isotonic', 
            cv=pds
        )        
        calibrated_model.fit(X_full, y_full)
        
        logger.info(f"Saving calibrated model to {save_path}")
        joblib.dump(calibrated_model, save_path)

    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    args = parse_args()
    train_and_calibrate(args)