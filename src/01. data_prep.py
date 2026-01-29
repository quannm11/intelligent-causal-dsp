import pandas as pd
import numpy as np
from datasets import Dataset
import os

DATA_PATH = "/accounts/masters/quannm/uplift_project/data/criteo_uplift.csv.gz"
SAVE_DIR = "/accounts/masters/quannm/uplift_project/data/v2_engineered"
SEED = 42

def engineer_features(df):
    print("Engineering features based on lift variance...")
    # Frequency Proxy
    df['user_freq'] = df.groupby('f0')['f0'].transform('count')
    
    # Quadratic Terms for Top Variance Features (f3=0.0044, f8=0.0037, f6=0.0023)
    for feat in ['f3', 'f8', 'f6']:
        df[f'{feat}_sq'] = df[feat] ** 2
        
    # Targeted Interaction
    df['f3_f6_inter'] = df['f3'] * df['f6']
    df['f2_f9_inter'] = df['f2'] * df['f9']
    return df

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(f"Reading CSV...")
    df = pd.read_csv(DATA_PATH)
    
    df = engineer_features(df)

    print("Performing 70/10/20 split (Train/Val/Test)...")
    train_df = df.sample(frac=0.7, random_state=SEED)
    rem_df = df.drop(train_df.index)
    
    # Split the remaining 30% into Val (1/3) and Test (2/3)
    val_df = rem_df.sample(frac=0.333, random_state=SEED)
    test_df = rem_df.drop(val_df.index)

    # Validation: Ensure conversion rates are balanced across splits
    for name, d in zip(["Train", "Val", "Test"], [train_df, val_df, test_df]):
        print(f"{name} Conv Rate: {d['conversion'].mean():.4%}")

    print("Saving to Arrow format...")
    Dataset.from_pandas(train_df).save_to_disk(os.path.join(SAVE_DIR, "train_data"))
    Dataset.from_pandas(val_df).save_to_disk(os.path.join(SAVE_DIR, "val_data"))
    Dataset.from_pandas(test_df).save_to_disk(os.path.join(SAVE_DIR, "test_data"))
    
    print(f"Done! Data versioned in {SAVE_DIR}")

if __name__ == "__main__":
    main()