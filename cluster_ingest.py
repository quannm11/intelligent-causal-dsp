import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datasets import load_dataset
import os
import time


OUTPUT_FILE = "criteo_uplift_full.parquet"
BATCH_SIZE = 500_000 
PRINT_INTERVAL = 5_000
total_rows = 0

start_time = time.time()
print(f"Starting Stream Processing -> {OUTPUT_FILE}")
os.environ["HF_HUB_TIMEOUT"] = "120" 

cache_path = "./data_cache" 
print("Downloading and caching dataset...")

dataset = load_dataset(
    "criteo/criteo-uplift", 
    split="train", 
    cache_dir=cache_path
)
writer = None
current_batch = []

for i, row in enumerate(dataset):
    current_batch.append(row)

    if (i + 1) % PRINT_INTERVAL == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        print(f"   ...Collected {i + 1} rows (Speed: {rate:.0f} rows/sec)", end="\r")
    
    if len(current_batch) >= BATCH_SIZE:
        df = pd.DataFrame(current_batch)
        table = pa.Table.from_pandas(df)
        
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_FILE, table.schema)
            
        writer.write_table(table)
        total_rows += len(current_batch)
        print(f"   Written {total_rows} rows...")
        current_batch = []

if current_batch:
    df = pd.DataFrame(current_batch)
    table = pa.Table.from_pandas(df)
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_FILE, table.schema)
    writer.write_table(table)

if writer:
    writer.close()

print(f" Done! Saved {total_rows} rows to {OUTPUT_FILE}")