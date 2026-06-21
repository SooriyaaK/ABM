import json

combos = [
    {"homophily": 0.0, "defector_frac": 0.0, "learning_rate": 0.0},   # all cooperators, no homophily
    {"homophily": 0.5, "defector_frac": 0.5, "learning_rate": 0.5},   # middle ground
    {"homophily": 1.0, "defector_frac": 1.0, "learning_rate": 1.0},   # all defectors, max homophily
    {"homophily": 0.0, "defector_frac": 1.0, "learning_rate": 0.5},   # all defectors, no homophily
    {"homophily": 1.0, "defector_frac": 0.0, "learning_rate": 0.5},   # all cooperators, max homophily
]

with open("params_test.json", "w") as f:
    json.dump(combos, f, indent=2)
print(f"Generated {len(combos)} parameter combinations → params_test.json")