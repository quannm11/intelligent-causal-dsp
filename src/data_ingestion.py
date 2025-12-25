import os
import shutil
import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher

LIMIT = 1_400_000        
BUFFER_SIZE = 100_000    
HASH_FEATURES = 1024     

OUTPUT_PATH = os.path.join("data", "processed")

NUM_COLS = ['f0', 'f2']
CAT_OHE_COLS = ['f1', 'f4', 'f5', 'f11']          # Low Cardinality
CAT_HASH_COLS = ['f3', 'f6', 'f7', 'f8', 'f9', 'f10'] # High Cardinality

def check_data_quality(df_chunk, scaler, hasher_output, chunk_idx):
    # 1. Sparsity Check 
    non_zeros = np.count_nonzero(hasher_output)
    total_cells = hasher_output.size
    sparsity = 1.0 - (non_zeros / total_cells)
    
    # 2. Distribution Shift Check 
    current_mean = df_chunk['f0'].mean()
    initial_mean = scaler.mean_[0] if hasattr(scaler, 'mean_') else 0
    
    denominator = initial_mean if abs(initial_mean) > 1e-9 else 1.0
    shift_pct = abs(current_mean - initial_mean) / denominator
    
    # Log every other chunk
    if chunk_idx % 2 == 0:
        print(f" [Monitor Chunk {chunk_idx}] Sparsity: {sparsity:.4f} | f0 Mean Shift: {shift_pct:.2%}")

def run_ingestion():
    
    # 0. Clean Setup
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Initialize Tools
    stream = load_dataset("criteo/criteo-uplift", split="train", streaming=True)
    
    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int8)
    hasher = FeatureHasher(n_features=HASH_FEATURES, input_type="string", dtype=np.float32)

    chunk_buffer = []
    total_rows_saved = 0
    chunk_index = 0
    is_first_chunk = True

    # 2. Step-sampling and Buffer
    for i, row in enumerate(stream):
        if total_rows_saved >= LIMIT:
            break
        
        if i % 10 == 0:
            chunk_buffer.append(row)

        if len(chunk_buffer) >= BUFFER_SIZE:
            df_chunk = pd.DataFrame(chunk_buffer)
            
            # First Chunk
            if is_first_chunk:
                print("   Fitting Scalers and Encoders on first chunk...")
                scaler.fit(df_chunk[NUM_COLS])
                ohe.fit(df_chunk[CAT_OHE_COLS])
                
                joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
                joblib.dump(ohe, os.path.join("models", "ohe.joblib"))
                is_first_chunk = False

            # Transformation
            # 1. Hashing 
            raw_strings = df_chunk[CAT_HASH_COLS].astype(str).values.tolist()
            hash_data = hasher.transform(raw_strings).toarray()
            
            # 2. Quality Checks 
            check_data_quality(df_chunk, scaler, hash_data, chunk_index)

            # 3. Scaling
            num_data = scaler.transform(df_chunk[NUM_COLS]).astype(np.float32)
            df_num = pd.DataFrame(num_data, columns=NUM_COLS, index=df_chunk.index)

            # 4. One-Hot Encoding
            ohe_data = ohe.transform(df_chunk[CAT_OHE_COLS])
            ohe_cols = ohe.get_feature_names_out(CAT_OHE_COLS)
            df_ohe = pd.DataFrame(ohe_data, columns=ohe_cols, index=df_chunk.index)

            # 5. Hash DataFrame
            hash_cols = [f'h_{j}' for j in range(HASH_FEATURES)]
            df_hash = pd.DataFrame(hash_data, columns=hash_cols, index=df_chunk.index)

            df_labels = df_chunk[['treatment', 'conversion', 'visit', 'exposure']].astype(np.int8)
            final_chunk = pd.concat([df_labels, df_num, df_ohe, df_hash], axis=1)

            # Write and Partition
            file_name = f"part_{chunk_index}.parquet"
            final_chunk.to_parquet(
                OUTPUT_PATH,
                partition_cols=['treatment'], 
                engine='pyarrow',
                index=False,
                basename_template=f"part_{chunk_index}_{{i}}.parquet"
            )

            total_rows_saved += len(chunk_buffer)
            print(f"   Saved Chunk {chunk_index}: Total {total_rows_saved} rows processed.")
            chunk_buffer = [] # Clear Buffer
            chunk_index += 1

    print(f"Ingestion Complete! Data saved to {OUTPUT_PATH}")
    
    # 3. Quick Sanity Check
    print("\n--- SANITY CHECK ---")
    try:
        check_path = os.path.join(OUTPUT_PATH, "treatment=1")
        check_df = pd.read_parquet(check_path)
        print(f"Sample Shape (Treatment=1): {check_df.shape}")
        print(f"Columns: {list(check_df.columns[:5])} ...")
    except Exception as e:
        print(f"Sanity check failed: {e}")

if __name__ == "__main__":
    run_ingestion()