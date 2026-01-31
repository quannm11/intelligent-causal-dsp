import numpy as np
import joblib
import pandas as pd
from src.config import T_MODEL_PATH, C_MODEL_PATH, CONVERSION_VALUE

class PIDBiddingAgent:
    def __init__(self, kp=0.1, ki=0.01, kd=0.05, target_spend_rate=0.1):
        """
        :param kp: Proportional gain (reacts to current error)
        :param ki: Integral gain (reacts to accumulated error)
        :param kd: Derivative gain (reacts to rate of change)
        :param target_spend_rate: The percentage of total budget we want to spend per time step
        """
        # Load Calibrated T-Learner
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
        self.adjustment_factor = 1.0  # Multiplier for the bid

    def update_controller(self, current_spend_rate):
        """Adjusts the bidding multiplier based on budget spend error."""
        error = self.target_rate - current_spend_rate
        
        self.integral_error += error
        derivative_error = error - self.last_error
        
        # Adjusts how aggressive the bid
        adjustment = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative_error)
        self.adjustment_factor = max(0.1, self.adjustment_factor + adjustment)
        
        self.last_error = error
        return self.adjustment_factor

    def predict_bid(self, features):
        """Calculates the final bid: Uplift * Conversion Value * PID Adjustment."""
        p_t = self.model_t.predict_proba(features)[:, 1]
        p_c = self.model_c.predict_proba(features)[:, 1]
        
        uplift = p_t - p_c
        
        # Causal Bidding Formula adjusted by PID
        raw_bid = uplift * CONVERSION_VALUE
        final_bid = np.maximum(0, raw_bid * self.adjustment_factor)
        
        return final_bid