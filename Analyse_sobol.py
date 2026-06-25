# analyse_sobol.py
import numpy as np, json
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import pandas as pd
import os

OUTPUT_FOLDER = "results_saltelli"
os.makedirs("sobol_output", exist_ok=True)

problem  = json.load(open(f"{OUTPUT_FOLDER}/saltelli_problem.json"))
Y_H      = np.load(f"{OUTPUT_FOLDER}/Y_H.npy")
Y_morans = np.load(f"{OUTPUT_FOLDER}/Y_morans.npy")
Y_nb_var = np.load(f"{OUTPUT_FOLDER}/Y_nb_var.npy")

outputs = {"H (segregation)": Y_H,
           "Moran's I":       Y_morans,
           "NB variance":     Y_nb_var}

# fig, axes = plt.subplots(len(outputs), 1, figsize=(8, 4 * len(outputs)))

# compute first, plot second
results = {}
for name, Y in outputs.items():
    results[name] = sobol.analyze(problem, Y, calc_second_order=False,
                                  print_to_console=True)


fig, axes = plt.subplots(len(outputs), 1, figsize=(8, 4 * len(outputs)))

for ax, (name, Si) in zip(axes, results.items()):
    names = problem["names"]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, Si["S1"], width, yerr=Si["S1_conf"],
           label="S1 (first-order)", color="tab:blue", capsize=4)
    ax.bar(x + width/2, Si["ST"], width, yerr=Si["ST_conf"],
           label="ST (total-order)", color="tab:orange", capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Sobol index")
    ax.set_ylim(0, 1)
    ax.set_title(f"Sobol indices — {name}")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig(f"sobol_output/sobol_indices.png", dpi=150)
plt.show()

# save CSV
rows = []
for name, Si in results.items():
    for i, param in enumerate(problem["names"]):
        rows.append({
            "output":    name,
            "parameter": param,
            "S1":        Si["S1"][i],
            "S1_conf":   Si["S1_conf"][i],
            "ST":        Si["ST"][i],
            "ST_conf":   Si["ST_conf"][i],
       })

pd.DataFrame(rows).to_csv(f"sobol_output/sobol_indices.csv", index=False)
print(f"Saved: sobol_output/sobol_indices.csv")

# scatter plots: each parameter vs each output
# useful for showing nonlinear relationships
# sadly lost input file so can't do it


# fig, axes = plt.subplots(len(outputs), len(problem["names"]), 
#                           figsize=(16, 12))
# param_values = np.load(f"{OUTPUT_FOLDER}/saltelli_problem_X.npy")

# for i, (name, Y) in enumerate(outputs.items()):
#     for j, param in enumerate(problem["names"]):
#         axes[i,j].scatter(param_values[:, j], Y, alpha=0.1, s=1)
#         axes[i,j].set_xlabel(param)
#         axes[i,j].set_ylabel(name)

# plt.tight_layout()
# plt.savefig(f"{OUTPUT_FOLDER}/scatter_plots.png", dpi=150)

# for ax, (name, Y) in zip(axes, outputs.items()):
#     Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=True)

#     names = problem["names"]
#     x = np.arange(len(names))
#     width = 0.35

#     ax.bar(x - width/2, Si["S1"],  width, yerr=Si["S1_conf"],
#            label="S1 (first-order)",  color="tab:blue",   capsize=4)
#     ax.bar(x + width/2, Si["ST"],  width, yerr=Si["ST_conf"],
#            label="ST (total-order)",  color="tab:orange",  capsize=4, alpha=0.8)

#     ax.set_xticks(x)
#     ax.set_xticklabels(names, rotation=15)
#     ax.set_ylabel("Sobol index")
#     ax.set_ylim(0, 1)
#     ax.set_title(f"Sobol indices — {name}")
#     ax.legend()
#     ax.axhline(0, color="black", linewidth=0.5)

# plt.tight_layout()
# plt.savefig(f"{OUTPUT_FOLDER}/sobol_indices.png", dpi=150)
# plt.show()