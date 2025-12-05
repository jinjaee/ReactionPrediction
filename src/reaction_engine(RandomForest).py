import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
import joblib
import os

class ReactionEngine:
    def __init__(self):
        print("Initializing Random Forest Engine...")
        
        # Paths to the files you downloaded from Colab
        model_path = "models/rf_model.pkl"
        feat_path = "models/magpie_featurizer.pkl"
        
        # Load the "Brain" and the "Translator" (Featurizer)
        if os.path.exists(model_path) and os.path.exists(feat_path):
            self.model = joblib.load(model_path)
            self.featurizer = joblib.load(feat_path)
            self.is_trained = True
            print("Random Forest Model Loaded Successfully.")
        else:
            print("WARNING: Models not found in 'models/' folder.")
            print("Please run the Colab script and move .pkl files here.")
            self.is_trained = False
        
    def generate_stoichiometry_grid(self, element_a, element_b):
        """Generates hypothetical formulas (A-B mixtures)"""
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
            # Fallback for testing API connections
            rng = np.random.default_rng()
            return rng.uniform(-3.0, 0.5, len(formulas)) 

        # 1. Convert strings (e.g. "Li2O") to Composition objects
        comps = []
        valid_indices = []
        for i, f in enumerate(formulas):
            try:
                comps.append(Composition(f))
                valid_indices.append(i)
            except:
                pass # Skip invalid formulas
        
        if not comps:
            return np.array([])

        # 2. Featurize
        # This uses the 'magpie_featurizer.pkl' to turn chemistry into math
        # ignore_errors=True prevents crashing on weird inputs
        features = self.featurizer.featurize_many(comps, ignore_errors=True)
        
        # Handle cases where featurization might fail (returns NaNs)
        features = np.array(features)
        # Simple clean: replace NaNs with 0 (rare edge case)
        features = np.nan_to_num(features)
        
        # 3. Predict using Random Forest
        pred_energies = self.model.predict(features)
        
        return pred_energies

    def get_reaction_products(self, element_a, element_b):
        # 1. Generate Candidates
        formulas = self.generate_stoichiometry_grid(element_a, element_b)
        
        # 2. Predict Energies
        energies = self.predict_energies(formulas)
        
        # 3. Construct Hull
        pd_entries = []
        # We need to be careful to match formulas to energies correctly
        # Re-convert formulas to compositions to match the loop
        valid_formulas = []
        for f in formulas:
            try:
                Composition(f)
                valid_formulas.append(f)
            except:
                pass

        if len(energies) != len(valid_formulas):
            # Fallback if sizes mismatch due to filtering
            # (In production, you'd align these more strictly)
            return [], None

        for formula, energy in zip(valid_formulas, energies):
            entry = PDEntry(Composition(formula), energy)
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
                    "formula": entry.composition.reduced_formula,
                    "energy_per_atom": entry.energy_per_atom,
                    "is_stable": True
                })
                
        return results, phase_diagram