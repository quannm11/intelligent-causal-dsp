import pandas as pd
import logging
import numpy as np
import os
from config import RAW_DATA_PATH, TRAIN_DATA, VAL_DATA, TEST_DATA, SEED, FEATURES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def engineer_features(df):
    logger.info("Starting feature engineering")
    
    try:
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
            logger.warning(f"The following features expected by the model are missing: {missing_cols}")
        else:
            logger.info(f"Feature engineering complete. All {len(FEATURES)} model features are present.")
            
        return df
        
    except KeyError as e:
        logger.error(f"Feature engineering failed. Missing column: {e}")
        raise

def main():
    logger.info(f"Checking data source at: {RAW_DATA_PATH}")
    if not os.path.exists(RAW_DATA_PATH):
        logger.error(f"File not found: {RAW_DATA_PATH}")
        return

    logger.info("Reading CSV")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Raw data loaded. Shape: {df.shape}")
    
    df = engineer_features(df)

    logger.info("Performing 70/10/20 split (Train/Val/Test)")
    train_df = df.sample(frac=0.7, random_state=SEED)
    rem_df = df.drop(train_df.index)
    
    val_df = rem_df.sample(frac=0.333, random_state=SEED)
    test_df = rem_df.drop(val_df.index)

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Saving parquet files to {TRAIN_DATA.parent}")
    try:
        train_df.to_parquet(TRAIN_DATA)
        val_df.to_parquet(VAL_DATA)
        test_df.to_parquet(TEST_DATA)
        logger.info("Data Prep Complete. Files saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save files: {e}")
        raise
    

if __name__ == "__main__":
    main()