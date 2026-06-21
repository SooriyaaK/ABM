"""
Small test grid for validating the pipeline end-to-end before the full run.
Run locally to generate params_test.json:
    python Params_test.py

Uses the CURRENT parameters (density, defector_frac, neighbourhood_count,
activation_rate) and covers the corners + middle of the space, with low/fast
values so a test array finishes quickly.
"""
import json

combos = [
    {"density": 0.3, "defector_frac": 0.0, "neighbourhood_count": 5,   "activation_rate": 0.5},  # sparse, all cooperators, few big nbhds
    {"density": 0.5, "defector_frac": 0.5, "neighbourhood_count": 25,  "activation_rate": 0.5},  # middle ground
    {"density": 0.8, "defector_frac": 1.0, "neighbourhood_count": 50,  "activation_rate": 1.0},  # dense, all defectors
    {"density": 0.8, "defector_frac": 0.0, "neighbourhood_count": 250, "activation_rate": 1.0},  # dense, all cooperators, max nbhds (heaviest case)
    {"density": 0.3, "defector_frac": 1.0, "neighbourhood_count": 5,   "activation_rate": 0.5},  # sparse, all defectors
]

with open("params_test.json", "w") as f:
    json.dump(combos, f, indent=2)

print(f"Generated {len(combos)} parameter combinations → params_test.json")
