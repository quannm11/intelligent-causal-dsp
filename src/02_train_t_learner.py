import os
import joblib
import pandas as pd
import xgboost as xgb
from datasets import load_from_disk
from sklearn.calibration import CalibratedClassifierCV

DATA_DIR = "/accounts/masters/quannm/uplift_project/data/v2_engineered"
MODEL_DIR = "/accounts/masters/quannm/uplift_project/models/v2"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading Training and Validation shards...")
train_df = load_from_disk(os.path.join(DATA_DIR, "train_data")).to_pandas()
val_df = load_from_disk(os.path.join(DATA_DIR, "val_data")).to_pandas()

features = [f'f{i}' for i in range(12)] + \
           ['user_freq', 'f3_sq', 'f8_sq', 'f6_sq', 'f3_f6_inter', 'f2_f9_inter']
target = 'conversion'

def train_and_calibrate():
    groups = [(1, "treatment"), (0, "control")]
    
    for group_id, label in groups:
        print(f"\n--- Processing {label.upper()} Model ---")
        
        X_train = train_df[train_df['treatment'] == group_id][features]
        y_train = train_df[train_df['treatment'] == group_id][target]
        X_val = val_df[val_df['treatment'] == group_id][features]
        y_val = val_df[val_df['treatment'] == group_id][target]
        
        base_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42
        )
        print(f"Fitting base XGBoost on Training Set")
        base_model.fit(X_train, y_train)
        
        print(f"Calibrating using dedicated Validation Set")
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model, 
            method='isotonic', 
            cv=None,           
            ensemble=False    
        )
        
        calibrated_model.fit(X_val, y_val)        
        model_path = os.path.join(MODEL_DIR, f"t_learner_{label}.joblib")
        joblib.dump(calibrated_model, model_path)
        print(f"Successfully saved calibrated model: {model_path}")

if __name__ == "__main__":
    train_and_calibrate()