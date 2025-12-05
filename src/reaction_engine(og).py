import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
# FIX: Correct import
from crabnet.crabnet_ import CrabNet

class ReactionEngine:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Engine initialized on device: {self.device}")
        
        print("Loading CrabNet model...")
        # Initialize the class
        self.model = CrabNet(mat_prop="EOH_formation_energy", compute_device=self.device)
        
        # Load the weights
        try:
            self.model.load_network("models/crabnet_model.pth")
            self.is_trained = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"WARNING: Could not load model. Error: {e}")
            print("Running in MOCK mode (Random Data).")
            self.is_trained = False
        
    def generate_stoichiometry_grid(self, element_a, element_b):
        hypothetical_formulas = []
        for x in range(1, 10):
            y = 10 - x
            hypothetical_formulas.append(f"{element_a}{x}{element_b}{y}")
        
        common_ratios = [(1,1), (1,2), (2,1), (2,3), (3,2), (2,5), (1,3)]
        for x, y in common_ratios:
             hypothetical_formulas.append(f"{element_a}{x}{element_b}{y}")
             
        return list(set(hypothetical_formulas))

    def predict_energies(self, formulas):
        if not self.is_trained:
            rng = np.random.default_rng()
            return rng.uniform(-3.0, 0.5, len(formulas)) 

        # Prepare DataFrame for CrabNet
        dummy_target = np.zeros(len(formulas))
        df = pd.DataFrame({"formula": formulas, "target": dummy_target})
        
        # Predict
        # CrabNet returns a tuple: (predictions, uncertainty, true_values)
        prediction_output = self.model.predict(df)
        
        # We only want the values (first item in tuple)
        if isinstance(prediction_output, tuple):
             pred = prediction_output[0]
        else:
             pred = prediction_output

        return pred.flatten()

    def get_reaction_products(self, element_a, element_b):
        formulas = self.generate_stoichiometry_grid(element_a, element_b)
        energies = self.predict_energies(formulas)
        
        pd_entries = []
        for formula, energy in zip(formulas, energies):
            try:
                comp = Composition(formula)
                entry = PDEntry(comp, energy)
                pd_entries.append(entry)
            except:
                continue
            
        pd_entries.append(PDEntry(Composition(element_a), 0.0))
        pd_entries.append(PDEntry(Composition(element_b), 0.0))
        
        phase_diagram = PhaseDiagram(pd_entries)
        stable_entries = phase_diagram.stable_entries
        
        results = []
        for entry in stable_entries:
            if entry.composition.reduced_formula not in [element_a, element_b]:
                results.append({
                    "formula": entry.composition.reduced_formula,
                    "energy_per_atom": entry.energy_per_atom,
                    "is_stable": True
                })
                
        return results, phase_diagram