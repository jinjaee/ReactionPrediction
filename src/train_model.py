import pandas as pd
import torch
import os
from crabnet.crabnet_ import CrabNet

def train():
    print("Loading training data...")
    train_df = pd.read_csv("data/train_data.csv")
    
    # --- DATA CLEANING ---
    initial_count = len(train_df)
    train_df = train_df.dropna(subset=["formula", "target"])
    train_df["formula"] = train_df["formula"].astype(str)
    train_df = train_df[~train_df["formula"].str.isnumeric()]
    print(f"Cleaned data: Removed {initial_count - len(train_df)} invalid rows.")
    # ---------------------

    val_df = train_df.sample(frac=0.1, random_state=42)
    train_df = train_df.drop(val_df.index)

    # --- THE FIX: FORCE CPU ---
    # The GPU driver cannot handle the sparse matrix operations in this library.
    device = torch.device("cpu")
    print(f"Training on device: {device} (Stable Mode)")

    # Initialize CrabNet
    # We use batch_size=128 which is optimized for CPU throughput
    model = CrabNet(
        mat_prop="EOH_formation_energy", 
        compute_device=device,
        batch_size=128 
    )

    # Train 
    print("Starting training... (Go grab a coffee, this will take ~1-2 hours)")
    model.fit(train_df=train_df, val_df=val_df)

    # Save
    os.makedirs("models", exist_ok=True)
    save_path = "models/crabnet_model.pth"
    model.save_network(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()