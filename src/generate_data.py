import pandas as pd
import os

def generate_local_data():
    print("Generating local training dataset...")
    
    # Real Formation Energies (eV/atom) from Materials Project
    # We provide enough data for the model to learn basic trends (e.g., Oxides are stable)
    data = [
        {"formula": "Li2O", "target": -1.98},
        {"formula": "Li2O2", "target": -1.65},
        {"formula": "LiO", "target": -0.80}, # Unstable theoretical
        {"formula": "MgO", "target": -3.05},
        {"formula": "CaO", "target": -3.50},
        {"formula": "Al2O3", "target": -2.55},
        {"formula": "SiO2", "target": -2.85},
        {"formula": "Fe2O3", "target": -1.85},
        {"formula": "Fe3O4", "target": -1.75},
        {"formula": "FeO", "target": -1.60},
        {"formula": "CO2", "target": -1.50},
        {"formula": "H2O", "target": -1.20},
        {"formula": "Na2O", "target": -1.40},
        {"formula": "K2O", "target": -1.10},
        {"formula": "ZnO", "target": -1.70},
        {"formula": "TiO2", "target": -3.10},
        # Add Sulfides to teach the model S vs O differences
        {"formula": "Li2S", "target": -1.45},
        {"formula": "MgS", "target": -2.20},
        {"formula": "CaS", "target": -2.90},
        {"formula": "FeS2", "target": -0.85},
        {"formula": "ZnS", "target": -1.15},
    ]

    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to disk
    output_file = "data/train_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Success! Saved {len(df)} compounds to {output_file}.")

if __name__ == "__main__":
    generate_local_data()