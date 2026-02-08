import numpy as np
import joblib
import pandas as pd

try:
    from src.config import T_MODEL_PATH, C_MODEL_PATH, CONVERSION_VALUE, FEATURES
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.config import T_MODEL_PATH, C_MODEL_PATH, CONVERSION_VALUE, FEATURES

class PIDBiddingAgent:
    def __init__(self, kp=0.1, ki=0.01, kd=0.05, target_spend_rate=0.1, integral_cap=10.0):
        # Load Calibrated T-Learner Models
        self.model_t = joblib.load(T_MODEL_PATH)
        self.model_c = joblib.load(C_MODEL_PATH)
        
        # PID Parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_rate = target_spend_rate
        
        # Controller State
        self.integral_error = 0
        self.last_error = 0
        self.adjustment_factor = 1.0
        self.integral_cap = integral_cap  # Prevent integral windup

    def update_controller(self, current_spend_rate):
        """Adjusts the bidding multiplier based on budget spend error."""
        error = self.target_rate - current_spend_rate
        self.integral_error += error
        
        self.integral_error = np.clip(self.integral_error, -self.integral_cap, self.integral_cap)
        
        derivative_error = error - self.last_error
        
        adjustment = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative_error)
        self.adjustment_factor = max(0.1, self.adjustment_factor + adjustment)
        
        self.last_error = error
        return self.adjustment_factor

    def predict_bid(self, input_data):
        """
        Calculates the bid.
        Accepts DataFrame or Dict. Handles feature alignment automatically.
        """
        # Standardize Input to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be a DataFrame or Dictionary")

        # Verify all features exist
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise KeyError(f"Input is missing features: {missing}")
            
        X = df[FEATURES].values

        # Predict Uplift
        p_t = self.model_t.predict_proba(X)[:, 1]
        p_c = self.model_c.predict_proba(X)[:, 1]
        
        uplift = p_t - p_c
        
        bid = np.maximum(0, uplift) * CONVERSION_VALUE * self.adjustment_factor
        
        return bid