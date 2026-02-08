import pandas as pd
import numpy as np
import os
from config import RAW_DATA_PATH, TRAIN_DATA, VAL_DATA, TEST_DATA, SEED, FEATURES

def engineer_features(df):
    print("Engineering features based on lift variance")
    
    # Frequency Proxy
    df['user_freq'] = df.groupby('f0')['f0'].transform('count')
    
    # Quadratic Terms
    for feat in ['f3', 'f8', 'f6']:
        df[f'{feat}_sq'] = df[feat] ** 2
        
    # Targeted Interactions
    df['f3_f6_inter'] = df['f3'] * df['f6']
    df['f2_f9_inter'] = df['f2'] * df['f9']
    
    missing_cols = [f for f in FEATURES if f not in df.columns]
    if missing_cols:
        print(f"Warning: The following model features are missing: {missing_cols}")
        
    return df

def main():
    print(f"Reading CSV from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    
    df = engineer_features(df)

    print("Performing 70/10/20 split (Train/Val/Test)")
    train_df = df.sample(frac=0.7, random_state=SEED)
    rem_df = df.drop(train_df.index)
    
    val_df = rem_df.sample(frac=0.333, random_state=SEED)
    test_df = rem_df.drop(val_df.index)

    print(f"Saving to {TRAIN_DATA.parent}")
    train_df.to_parquet(TRAIN_DATA)
    val_df.to_parquet(VAL_DATA)
    test_df.to_parquet(TEST_DATA)
    

if __name__ == "__main__":
    main()