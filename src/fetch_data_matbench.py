import pandas as pd
from matminer.datasets import load_dataset
import os

def download_matbench_data():
    print("Downloading 'matbench_mp_e_form' (Materials Project Formation Energy)...")
    print("This dataset contains ~132,000 samples. This may take a minute.")
    
    # Load the official benchmark dataset (bypasses mp-api)
    df = load_dataset("matbench_mp_e_form")
    
    # The dataset comes with 'structure' objects. 
    # We need to extract the text Formula for CrabNet.
    print("Converting crystal structures to formulas...")
    
    # Helper function to get formula from pymatgen Structure
    def get_formula(structure):
        return structure.composition.reduced_formula

    df["formula"] = df["structure"].apply(get_formula)
    
    # Rename the energy column to 'target' for CrabNet
    df["target"] = df["e_form"]
    
    # Keep only the columns we need
    clean_df = df[["formula", "target"]]
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/train_data.csv"
    clean_df.to_csv(output_path, index=False)
    
    print(f"Success! Saved {len(clean_df)} compounds to {output_path}.")
    print("You can now train CrabNet on this massive dataset.")

if __name__ == "__main__":
    download_matbench_data()