import pandas as pd
import torch
import time
from crabnet.crabnet_ import CrabNet

def test_cpu_speed():
    print("Loading data for TIMING TEST...")
    train_df = pd.read_csv("data/train_data.csv")
    
    # Constants for Prediction
    REAL_DATA_SIZE = len(train_df) # Should be ~132,000
    REAL_EPOCHS = 300
    
    TEST_DATA_SIZE = 500
    TEST_EPOCHS = 2
    
    # --- MINI DATASET ---
    train_df = train_df.head(TEST_DATA_SIZE)
    
    # Clean data 
    train_df = train_df.dropna(subset=["formula", "target"])
    train_df["formula"] = train_df["formula"].astype(str)
    train_df = train_df[~train_df["formula"].str.isnumeric()]

    val_df = train_df.sample(frac=0.1)
    train_df = train_df.drop(val_df.index)

    device = torch.device("cpu")
    print(f"Testing on device: {device}")

    # Initialize model
    # CRITICAL: We use batch_size=128 to match the real training script
    # This ensures our speed estimate is realistic.
    model = CrabNet(
        mat_prop="EOH_timing_test", 
        compute_device=device, 
        batch_size=128,
        epochs=TEST_EPOCHS
    )

    print(f"\n--- STARTING TIMER ({TEST_EPOCHS} Epochs, {TEST_DATA_SIZE} Samples) ---")
    start_time = time.time()
    
    model.fit(train_df=train_df, val_df=val_df)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"--- TIMING COMPLETE ---\n")
    print(f"Test Run Time: {elapsed:.2f} seconds")
    
    # --- PREDICTION MATH ---
    # 1. Scale by Data Size (120,000 / 500)
    data_scale = REAL_DATA_SIZE / TEST_DATA_SIZE
    
    # 2. Scale by Epochs (300 / 2)
    epoch_scale = REAL_EPOCHS / TEST_EPOCHS
    
    # 3. Total Linear Projection
    estimated_seconds = elapsed * data_scale * epoch_scale
    
    # Note: Real training is often faster per-sample than test training 
    # because overhead (loading/graph generation) is amortized over longer runs.
    # We apply a conservative 0.7 correction factor for overhead removal.
    likely_seconds = estimated_seconds * 0.7
    
    est_minutes = likely_seconds / 60
    est_hours = est_minutes / 60
    
    print(f"--- ESTIMATE FOR FULL RUN ---")
    print(f"Scaling Factors: {data_scale:.1f}x (Data) * {epoch_scale}x (Epochs)")
    print(f"Raw Linear Projection: {estimated_seconds/3600:.2f} hours")
    print(f"Adjusted Estimate (removing overhead): ~{est_hours:.2f} hours")
    print("-----------------------------")

if __name__ == "__main__":
    test_cpu_speed()