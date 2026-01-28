import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import os

DATA_PATH = "/accounts/masters/quannm/uplift_project/data/criteo_uplift.csv.gz"
SAVE_DIR = "/accounts/masters/quannm/uplift_project/data"
SEED = 42

def main():
    print(f"Reading CSV into memory")
    df = pd.read_csv(DATA_PATH, engine='c')
    print(f"   -> Loaded {len(df):,} rows.")

    print("Performing 80/20 split...")
    df_train = df.sample(frac=0.8, random_state=SEED)
    df_test = df.drop(df_train.index)

    print("-" * 30)
    train_conv = df_train['conversion'].mean()
    test_conv = df_test['conversion'].mean()
    print(f"Train Conversion Rate: {train_conv:.4%}")
    print(f"Test Conversion Rate:  {test_conv:.4%}")
    print("-" * 30)

    print("Converting to Arrow and saving to disk...")
    train_ds = Dataset.from_pandas(df_train)
    test_ds = Dataset.from_pandas(df_test)
    
    train_ds.save_to_disk(os.path.join(SAVE_DIR, "train_data"))
    test_ds.save_to_disk(os.path.join(SAVE_DIR, "test_data"))
    
    print("Done! Check your /data folder.")

if __name__ == "__main__":
    main()