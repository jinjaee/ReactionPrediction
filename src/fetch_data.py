import pandas as pd
from mp_api.client import MPRester

# !!! REPLACE WITH YOUR ACTUAL API KEY !!!
MAPI_KEY = "sMtW6BD2WSHpN2tqUK0ySKE7Ogjcih5B"

def download_training_data():
    print("Connecting to Materials Project...")
    
    # Initialize connection
    with MPRester(MAPI_KEY) as mpr:
        # FIX 1: Use 'mpr.materials.summary' instead of 'mpr.summary'
        # FIX 2: num_elements must be a tuple (min, max). We want exactly 2.
        docs = mpr.materials.summary.search(
            num_elements=(2, 2), 
            fields=["formula_pretty", "formation_energy_per_atom"]
        )

    print(f"Downloaded {len(docs)} compounds.")

    # Format into the CSV structure CrabNet expects
    data = []
    for doc in docs:
        data.append({
            "formula": doc.formula_pretty,
            "target": doc.formation_energy_per_atom
        })

    df = pd.DataFrame(data)
    
    # Save to disk
    output_file = "train_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}. Ready for training.")

if __name__ == "__main__":
    download_training_data()