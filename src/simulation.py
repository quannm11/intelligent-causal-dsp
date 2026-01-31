import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import TEST_DATA, T_MODEL_PATH, C_MODEL_PATH, CONVERSION_VALUE
import joblib

class OfflineSimulator:
    def __init__(self, budget=500.0):
        self.budget = budget
        self.model_t = joblib.load(T_MODEL_PATH)
        self.model_c = joblib.load(C_MODEL_PATH)
        
    def run_comparison(self, data):
        """Simulates bidding across the test set."""
        results = []
        # features = list of your engineered features
        features = [f for f in data.columns if f.startswith('f')] 
        
        # 1. Standard Agent (Bids on P(Y|T=1) only)
        p_buy = self.model_t.predict_proba(data[features])[:, 1]
        
        # 2. Uplift Agent (Bids on P(Y|T=1) - P(Y|T=0))
        p_ctrl = self.model_c.predict_proba(data[features])[:, 1]
        uplift = p_buy - p_ctrl
        
        # Calculate Bids
        data['bid_propensity'] = p_buy * CONVERSION_VALUE
        data['bid_uplift'] = np.maximum(0, uplift * CONVERSION_VALUE)
        
        # Evaluation: We only "win" and "see" outcomes if our bid is high
        # In this simple offline version, we sort by bid and see cumulative conversions
        for strategy in ['bid_propensity', 'bid_uplift']:
            temp_df = data.sort_values(by=strategy, ascending=False).copy()
            # Calculate cumulative incremental gain (Uplift)
            temp_df['cum_conversions'] = temp_df['treatment_effect_label'].cumsum() 
            results.append(temp_df['cum_conversions'].values)
            
        return results

# --- Run the Simulation ---
if __name__ == "__main__":
    from datasets import load_from_disk
    test_ds = load_from_disk(TEST_DATA).to_pandas().sample(50000)
    
    sim = OfflineSimulator()
    propensity_res, uplift_res = sim.run_comparison(test_ds)
    
    plt.figure(figsize=(10, 6))
    plt.plot(uplift_res, label='Causal Uplift Agent (Your Model)', color='green', linewidth=2)
    plt.plot(propensity_res, label='Propensity Agent (Baseline)', color='gray', linestyle='--')
    plt.title("The Showdown: Causal Bidding vs. Propensity Bidding")
    plt.xlabel("Ad Impressions (Ranked by Bid)")
    plt.ylabel("Cumulative Incremental Conversions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("results/the_showdown.png")
    print("Simulation complete. Chart saved to results/the_showdown.png")