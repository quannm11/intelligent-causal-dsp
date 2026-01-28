import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_qini(df):
    """
    Calculates the Qini curve values.
    """
    # 1. Rank users by uplift score (highest to lowest)
    df = df.sort_values('uplift_score', ascending=False).reset_index(drop=True)
    
    # 2. Calculate cumulative counts
    df['n_t'] = (df['treatment'] == 1).cumsum()
    df['n_c'] = (df['treatment'] == 0).cumsum()
    
    df['y_t'] = ((df['treatment'] == 1) & (df['conversion'] == 1)).cumsum()
    df['y_c'] = ((df['treatment'] == 0) & (df['conversion'] == 1)).cumsum()
    
    # 3. Qini Formula: y_t - (y_c * n_t / n_c)
    # This adjusts for the fact that group sizes (T vs C) are unequal
    df['qini'] = df['y_t'] - (df['y_c'] * df['n_t'] / df['n_c'])
    
    # Fill NaN (happens at the very start if n_c is 0)
    df['qini'] = df['qini'].fillna(0)
    
    return df

# Load the predictions you just saved
print("Loading predictions...")
test_df = pd.read_csv("data/test_predictions.csv")

# Run calculation
qini_df = calculate_qini(test_df)

# 4. Plotting
print("Generating Qini Curve...")
plt.figure(figsize=(10, 6))

# The Model Curve
plt.plot(qini_df['qini'].values, label='T-Learner (XGBoost)', color='blue')

# The Random Baseline (straight line from 0 to the final Qini value)
plt.plot([0, len(qini_df)], [0, qini_df['qini'].iloc[-1]], 
         'r--', label='Random Guessing')

plt.title('Qini Curve - Criteo Uplift')
plt.xlabel('Number of Users Targeted (Ranked by Score)')
plt.ylabel('Cumulative Incremental Conversions')
plt.legend()
plt.grid(True)
plt.savefig("data/qini_curve.png")

# 5. Calculate AUUC (Area Under Uplift Curve)
# Higher is better.
auuc = np.trapz(qini_df['qini'].values) / (len(qini_df) * qini_df['qini'].iloc[-1])
print(f"AUUC Score: {auuc:.4f}")
print("Plot saved to data/qini_curve.png")
