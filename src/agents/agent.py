import numpy as np
import joblib
import pandas as pd
from 00_config import T_MODEL_PATH, C_MODEL_PATH, CONVERSION_VALUE, FEATURES

class PIDBiddingAgent:
    def __init__(self, kp=0.1, ki=0.01, kd=0.05, target_spend_rate=0.1):
        # Load Models
        self.model_t = joblib.load(T_MODEL_PATH)
        self.model_c = joblib.load(C_MODEL_PATH)
        
        # PID Parameters
        self.kp, self.ki, self.kd = kp, ki, kd
        self.target_rate = target_spend_rate
        
        # Controller State
        self.integral_error = 0
        self.last_error = 0
        self.adjustment_factor = 1.0 

    def predict_bid(self, df_features):
        """Calculates bid based on Causal Uplift."""
        X = df_features[FEATURES].values
        
        p_t = self.model_t.predict_proba(X)[:, 1]
        p_c = self.model_c.predict_proba(X)[:, 1]
        
        uplift = p_t - p_c
        
        bid = np.maximum(0, uplift) * CONVERSION_VALUE * self.adjustment_factor
        return bid