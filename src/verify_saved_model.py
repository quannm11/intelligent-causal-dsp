import joblib
import pandas as pd
import numpy as np
import os

# 1. Path check
models_path = {
    "Treatment": "models/t_learner_treatment.joblib",
    "Control": "models/t_learner_control.joblib"
}

def verify():
    print("--- Model Persistence Verification ---")
    
    for name, path in models_path.items():
        # Check if file exists
        if not os.path.exists(path):
            print(f"❌ ERROR: {path} not found!")
            continue
            
        print(f"Loading {name} model...")
        model = joblib.load(path)
        
        # Verify it's an XGBoost model
        print(f"✅ Loaded {type(model).__name__}")
        
        # 2. Test Prediction (Smoke Test)
        # Create a dummy row with 12 features (f0-f11)
        dummy_data = pd.DataFrame(np.random.rand(1, 12), columns=[f'f{i}' for i in range(12)])
        
        try:
            prob = model.predict_proba(dummy_data)[:, 1][0]
            print(f"✅ Prediction test successful! Output: {prob:.6f}")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    verify()
