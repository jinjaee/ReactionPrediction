import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
import joblib
import os

# --- 1. DEFINE THE NETWORK (Must match the trained model exactly) ---
class MagpieNet(nn.Module):
    def __init__(self, input_dim):
        super(MagpieNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.network(x)

# --- 2. THE ENGINE CLASS ---
class ReactionEngine:
    def __init__(self):
        print("Initializing Neural Network Engine...")
        # Use Mac GPU (mps) if available, otherwise CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Running on: {self.device}")
        
        # Paths to your new files
        model_path = "models/mlp_model.pth"
        feat_path = "models/magpie_featurizer.pkl"
        dim_path = "models/input_dim.pkl"
        
        if os.path.exists(model_path) and os.path.exists(feat_path):
            # Load the helper files
            self.featurizer = joblib.load(feat_path)
            input_dim = joblib.load(dim_path)
            
            # Rebuild the Network Architecture
            self.model = MagpieNet(input_dim).to(self.device)
            
            # Load the trained "Brain" weights
            # map_location ensures it loads on Mac even if trained on NVIDIA
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() # Set to "Thinking Mode" (not Training Mode)
            
            self.is_trained = True
            print("Neural Network Loaded Successfully.")
        else:
            print("WARNING: Model files missing in 'models/' folder.")
            self.is_trained = False
        
    def generate_stoichiometry_grid(self, element_a, element_b):
        # Generate ratios like A1B1, A1B2, A2B3...
        hypothetical_formulas = []
        for x in range(1, 10):
            y = 10 - x
            hypothetical_formulas.append(f"{element_a}{x}{element_b}{y}")
        
        common = [(1,1), (1,2), (2,1), (2,3), (3,2), (2,5), (1,3)]
        for x, y in common:
             hypothetical_formulas.append(f"{element_a}{x}{element_b}{y}")
             
        return list(set(hypothetical_formulas))

    def predict_energies(self, formulas):
        if not self.is_trained:
            return np.random.uniform(-3.0, 0.5, len(formulas)) 

        # 1. Convert Text -> Chemistry Objects
        comps = []
        for f in formulas:
            try:
                comps.append(Composition(f))
            except:
                pass
        
        if not comps: return np.array([])

        # 2. Featurize (Convert Chemistry -> Math Vectors)
        # This runs on CPU
        features = self.featurizer.featurize_many(comps, ignore_errors=True)
        
        # Clean up any bad data (NaNs) -> Convert to Float32 for PyTorch
        features = np.nan_to_num(np.array(features)).astype(np.float32)
        
        # 3. Predict (Convert Math -> Energy)
        # This runs on your Mac GPU (mps)
        with torch.no_grad():
            tensor_X = torch.tensor(features).to(self.device)
            preds = self.model(tensor_X)
            
        # Return as a simple list of numbers
        return preds.cpu().numpy().flatten()

    def get_reaction_products(self, element_a, element_b):
        # 1. Generate Candidates
        formulas = self.generate_stoichiometry_grid(element_a, element_b)
        
        # 2. Predict Energies
        energies = self.predict_energies(formulas)
        
        # 3. Build Phase Diagram (Convex Hull)
        pd_entries = []
        valid_formulas = [f for f in formulas if Composition(f)]
        
        if len(energies) != len(valid_formulas):
            return [], None

        for formula, energy in zip(valid_formulas, energies):
            # --- CRITICAL FIX: Convert Numpy float to Python float ---
            safe_energy = float(energy) 
            entry = PDEntry(Composition(formula), safe_energy)
            pd_entries.append(entry)
            
        # Add Pure Elements (Reference States = 0.0)
        pd_entries.append(PDEntry(Composition(element_a), 0.0))
        pd_entries.append(PDEntry(Composition(element_b), 0.0))
        
        phase_diagram = PhaseDiagram(pd_entries)
        stable_entries = phase_diagram.stable_entries
        
        results = []
        for entry in stable_entries:
            if entry.composition.reduced_formula not in [element_a, element_b]:
                results.append({
                    "formula": str(entry.composition.reduced_formula), # Ensure String
                    "energy_per_atom": float(entry.energy_per_atom),   # Ensure Float
                    "is_stable": True
                })
        return results, phase_diagram