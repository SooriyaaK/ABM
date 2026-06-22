# collect_outputs.py
import numpy as np, json, glob

OUTPUT_FOLDER = "results_saltelli"

# open the files; sort by combo index
files = sorted(glob.glob(f"{OUTPUT_FOLDER}/run_*.npz"),
               key=lambda f: int(f.split("combo")[1].split("_")[0]))


# One scalar per combo — average over seeds
Y_H         = []   # mean final segregation H
Y_morans    = []   # mean Moran's I
Y_nb_var    = []   # mean neighbourhood income variance
Y_steps     = []   # mean convergence steps

for f in files:
    d = np.load(f, allow_pickle=True)
    Y_H.append(d["all_H"][:, -1].mean())          # last timestep, mean over seeds (collapse to single scalar)
    Y_morans.append(d["final_morans_I"].mean())
    Y_nb_var.append(d["final_nb_variance"].mean())
    Y_steps.append(d["convergence_steps"].mean())

# save as np array for sobol indices calc
# these are what analyse_sobol.py will consume
np.save(f"{OUTPUT_FOLDER}/Y_H.npy",      Y_H) 
np.save(f"{OUTPUT_FOLDER}/Y_morans.npy", Y_morans)
np.save(f"{OUTPUT_FOLDER}/Y_nb_var.npy", Y_nb_var)
np.save(f"{OUTPUT_FOLDER}/Y_steps.npy",  Y_steps)

print(f"Collected {len(Y_H)} combos")