import os
import joblib
import pandas as pd
from datasets import load_from_disk
import numpy as np

DATA_DIR = "/accounts/masters/quannm/uplift_project/data/v2_engineered"
MODEL_DIR = "/accounts/masters/quannm/uplift_project/models/v2"
OUTPUT_DIR = "/accounts/masters/quannm/uplift_project/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Test Data and Calibrated Models...")
test_df = load_from_disk(os.path.join(DATA_DIR, "test_data")).to_pandas()
model_t = joblib.load(os.path.join(MODEL_DIR, "t_learner_treatment.joblib"))
model_c = joblib.load(os.path.join(MODEL_DIR, "t_learner_control.joblib"))

features = [f'f{i}' for i in range(12)] + \
           ['user_freq', 'f3_sq', 'f8_sq', 'f6_sq', 'f3_f6_inter', 'f2_f9_inter']

def run_inference():
    print("Calculating Calibrated Probabilities")
    prob_t = model_t.predict_proba(test_df[features])[:, 1]
    prob_c = model_c.predict_proba(test_df[features])[:, 1]
    
    # Calculate Uplift Score (CATE)
    test_df['uplift_score'] = prob_t - prob_c
    
    output_path = os.path.join(OUTPUT_DIR, "final_test_predictions.parquet")
    test_df.to_parquet(output_path, index=False)
    print(f"Inference complete. Scores saved to: {output_path}")
    
if __name__ == "__main__":
    run_inference()