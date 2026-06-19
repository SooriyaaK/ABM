"""
Run this once locally to generate params.json before submitting the array job:
    python params.py
"""
import json
import numpy as np

# the values we want to vary with (min, max, number)
homophily_values     = np.linspace(0, 1, 10).tolist()
defector_frac_values = np.linspace(0, 1, 10).tolist()
learning_rate_values = np.linspace(0, 1, 10).tolist()

combos = [
    {"homophily": round(h, 4), "defector_frac": round(d, 4), "learning_rate": round(l, 4)}
    for h in homophily_values
    for d in defector_frac_values
    for l in learning_rate_values
]

with open("params.json", "w") as f:
    json.dump(combos, f, indent=2)

print(f"Generated {len(combos)} parameter combinations → params.json")
