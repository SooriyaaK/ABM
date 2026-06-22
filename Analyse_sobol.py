# analyse_sobol.py
import numpy as np, json
from SALib.analyze import sobol
import matplotlib.pyplot as plt

OUTPUT_FOLDER = "results_saltelli"

problem  = json.load(open(f"{OUTPUT_FOLDER}/saltelli_problem.json"))
Y_H      = np.load(f"{OUTPUT_FOLDER}/Y_H.npy")
Y_morans = np.load(f"{OUTPUT_FOLDER}/Y_morans.npy")
Y_nb_var = np.load(f"{OUTPUT_FOLDER}/Y_nb_var.npy")

outputs = {"H (segregation)": Y_H,
           "Moran's I":       Y_morans,
           "NB variance":     Y_nb_var}

fig, axes = plt.subplots(len(outputs), 1, figsize=(8, 4 * len(outputs)))

for ax, (name, Y) in zip(axes, outputs.items()):
    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=True)

    names = problem["names"]
    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, Si["S1"],  width, yerr=Si["S1_conf"],
           label="S1 (first-order)",  color="tab:blue",   capsize=4)
    ax.bar(x + width/2, Si["ST"],  width, yerr=Si["ST_conf"],
           label="ST (total-order)",  color="tab:orange",  capsize=4, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Sobol index")
    ax.set_ylim(0, 1)
    ax.set_title(f"Sobol indices — {name}")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/sobol_indices.png", dpi=150)
plt.show()