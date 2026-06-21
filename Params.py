"""
Run this once locally to generate params.json before submitting the array job:
    python params.py
"""
import json
import numpy as np

# the values we want to vary with (min, max, number)
density_values     = np.linspace(0, 1, 10).tolist()
defector_frac_values = np.linspace(0, 1, 10).tolist()
neighbourhood_count_values = np.linspace(5, 200, 10).tolist()
activation_rate_values = np.linspace(0, 1, 10).tolist()

combos = [
    {"density": round(h, 4), "defector_frac": round(d, 4), "neighbourhood_count": int(l), "activation_rate": round(s, 4)}
    for h in density_values
    for d in defector_frac_values
    for l in neighbourhood_count_values
    for s in activation_rate_values
]

with open("params.json", "w") as f:
    json.dump(combos, f, indent=2)

print(f"Generated {len(combos)} parameter combinations → params.json")
