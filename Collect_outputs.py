# collect_outputs.py
import numpy as np, json, glob

OUTPUT_FOLDER = "results_saltelli"

# open the files; sort by combo index
files = sorted(glob.glob(f"{OUTPUT_FOLDER}/run_*.npz"),
               key=lambda f: int(f.split("combo")[1].split("_")[0]))

file_dict = {}
for f in files:
    idx = int(f.split("combo")[1].split("_")[0])
    file_dict[idx] = f

# One scalar per combo — average over seeds
Y_H         = []   # mean final segregation H
Y_morans    = []   # mean Moran's I
Y_nb_var    = []   # mean neighbourhood income variance
Y_steps     = []   # mean convergence steps

for i in range(6144):
    if i in file_dict:
        d = np.load(file_dict[i], allow_pickle=True) # pickle means 'I trust this file'
        Y_H.append(d["all_H"][:, -1].mean())
        
        # handle None values in morans and nb_var; prob from near-empty neighbourhoods
        morans = d["final_morans_I"]
        morans = np.array([x for x in morans if x is not None], dtype=float)
        Y_morans.append(morans.mean() if len(morans) > 0 else np.nan)
        
        nb_var = d["final_nb_variance"]
        nb_var = np.array([x for x in nb_var if x is not None], dtype=float)
        Y_nb_var.append(nb_var.mean() if len(nb_var) > 0 else np.nan)
        
        Y_steps.append(d["convergence_steps"].mean())
    else:
        print(f"Missing combo {i} — filling with NaN")
        Y_H.append(np.nan)
        Y_morans.append(np.nan)
        Y_nb_var.append(np.nan)
        Y_steps.append(np.nan)

# for f in files:
#     d = np.load(f, allow_pickle=True)
#     Y_H.append(d["all_H"][:, -1].mean())          # last timestep, mean over seeds (collapse to single scalar)
#     Y_morans.append(d["final_morans_I"].mean())
#     Y_nb_var.append(d["final_nb_variance"].mean())
#     Y_steps.append(d["convergence_steps"].mean())

# fill NaN with mean of the rest
Y_H      = np.array(Y_H);      Y_H[np.isnan(Y_H)]           = np.nanmean(Y_H)
Y_morans = np.array(Y_morans); Y_morans[np.isnan(Y_morans)]  = np.nanmean(Y_morans)
Y_nb_var = np.array(Y_nb_var); Y_nb_var[np.isnan(Y_nb_var)]  = np.nanmean(Y_nb_var)
Y_steps  = np.array(Y_steps);  Y_steps[np.isnan(Y_steps)]    = np.nanmean(Y_steps)


# save as np array for sobol indices calc
# these are what analyse_sobol.py will consume
np.save(f"{OUTPUT_FOLDER}/Y_H.npy",      Y_H) 
np.save(f"{OUTPUT_FOLDER}/Y_morans.npy", Y_morans)
np.save(f"{OUTPUT_FOLDER}/Y_nb_var.npy", Y_nb_var)
np.save(f"{OUTPUT_FOLDER}/Y_steps.npy",  Y_steps)

print(f"Collected {len(Y_H)} combos")