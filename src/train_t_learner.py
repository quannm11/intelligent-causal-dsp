import os
import pandas as pd
import xgboost as xgb
from datasets import load_from_disk
from sklearn.metrics import log_loss

print("Loading pre-split Arrow data...")
train_ds = load_from_disk("data/train_data")
test_ds = load_from_disk("data/test_data")

train_df = train_ds.to_pandas()
test_df = test_ds.to_pandas()

#drop 'visit' and 'exposure'
features = [f'f{i}' for i in range(12)]
target = 'conversion'

# Split by Treatment
df_t = train_df[train_df['treatment'] == 1]
df_c = train_df[train_df['treatment'] == 0]

print(f"Training on {len(df_t):,} Treatment samples and {len(df_c):,} Control samples...")

# Hyperparameters: Using small values for quick training on cluster
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'tree_method': 'hist', 
    'random_state': 42
}
model_t = xgb.XGBClassifier(**params)
model_c = xgb.XGBClassifier(**params)

print("Training Treatment Model...")
model_t.fit(df_t[features], df_t[target])

print("Training Control Model...")
model_c.fit(df_c[features], df_c[target])

print("\nCalculating Uplift on Test Set...")
# P(Y|T=1)
prob_t = model_t.predict_proba(test_df[features])[:, 1]
# P(Y|T=0)
prob_c = model_c.predict_proba(test_df[features])[:, 1]

# Uplift Score = P(Y|T=1) - P(Y|T=0)
test_df['uplift_score'] = prob_t - prob_c

print("\nModel trained! First 5 Uplift Scores:")
print(test_df[['uplift_score']].head())

# Save results for evaluation
test_df.to_csv("data/test_predictions.csv", index=False)

import joblib
import os

# Create directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the models while they are still in memory
joblib.dump(model_t, "models/t_learner_treatment.joblib")
joblib.dump(model_c, "models/t_learner_control.joblib")

print("Models saved successfully!")