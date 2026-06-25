#
#
#

# generate_saltelli_params.py
import json
from SALib.sample import sobol as saltelli  # same API, new name
import numpy as np
import os
os.makedirs("results_saltelli", exist_ok=True)

# problem dict
problem = {
    'num_vars': 4, # number of params we are varying
    'names': ['density', 'defector_frac', 'neighbourhood_count', 'activation_rate'], # labels for plots
    'bounds': [ # samples uniformly between these bounds
        [0.0,  1.0],   # density
        [0.0,  1.0],   # defector_frac
        [5,   200],     # neighbourhood_count (treat as continuous, round in model)
        [0.0,  1.0],   # activation_rate
    ]
}

# N = base sample size. Total runs = N * (2*num_vars + 2) = N * 10
# For N=1000 → 10,000 combos (same as before)
N = 1024 # closest power of 2
param_values = saltelli.sample(problem, N, calc_second_order=False) # creates those A, B, A_B^i matrices

# convert to json format
combos = [
    {
        "density":              float(row[0]),
        "defector_frac":        float(row[1]),
        "neighbourhood_count":  int(round(row[2])), # SALib gives floats, model needs int
        "activation_rate":      float(row[3]),
    }
    for row in param_values
]

with open("results_saltelli/saltelli_params.json", "w") as f:
    json.dump(combos, f, indent=2)

print(f"Generated {len(combos)} combos")  # → 10,000

# Also save the problem + raw param_values for analysis later

# shape (6000, 4) numpy array (the actual numbers SALib generated)
# needed later so sobol.analyze() knows the exact sample structure
np.save("results_saltelli/saltelli_problem_X.npy", param_values) # raw array; X for 'input'

# the dict with num_vars, names, bounds
# needed later so sobol.analyze() knows what the columns mean
json.dump(problem, open("results_saltelli/saltelli_problem.json", "w")) # bounds and names