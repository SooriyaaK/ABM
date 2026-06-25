import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results_summary.csv")
os.makedirs("sobol_output", exist_ok=True)
OUTPUT_FOLDER = "sobol_output"
NB = 12  # ~6000 samples / 144 bins ≈ 40 per bin
edges = np.linspace(0, 1, NB + 1)
df["x_bin"] = pd.cut(df["activation_rate"], edges)
df["y_bin"] = pd.cut(df["density"],         edges)

g = df.groupby(["y_bin", "x_bin"], observed=False)["H_tail_mean"]
mean_grid, count_grid = g.mean().unstack(), g.size().unstack()
masked = np.ma.masked_where(count_grid.values < 5, mean_grid.values)

fig, ax = plt.subplots(figsize=(6.5, 5))
cmap = plt.cm.viridis.copy()
cmap.set_bad("lightgray")
im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
ax.set_xlabel("activation rate") 
ax.set_ylabel("agent density")
ax.set_title("Final segregation (H) heatmap over the activation rate \n and agent density")
fig.colorbar(im, ax=ax, label="H (tail mean)")
plt.savefig(f"{OUTPUT_FOLDER}/sa_heatmap.png", dpi=150)
plt.close()

plt.scatter(df.activation_rate, df.H_tail_mean, c=df.density, s=6)
plt.xlabel("activation rate")
plt.ylabel("H tail mean")
plt.title("Final segregation (H) scatter plot over the activation rate")
plt.savefig(f"{OUTPUT_FOLDER}/sa_scatter.png", dpi=150)



nc_edges = np.linspace(df["neighbourhood_count"].min(),
                       df["neighbourhood_count"].max(), NB + 1)   # ~5 to 250
df["x_bin"] = pd.cut(df["neighbourhood_count"], nc_edges)
df["y_bin"] = pd.cut(df["density"],         edges)

g = df.groupby(["y_bin", "x_bin"], observed=False)["morans_I_mean"]
mean_grid, count_grid = g.mean().unstack(), g.size().unstack()
masked = np.ma.masked_where(count_grid.values < 5, mean_grid.values)

fig, ax = plt.subplots(figsize=(6.5, 5))
cmap = plt.cm.viridis.copy()
cmap.set_bad("lightgray")
im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap,
               extent=[nc_edges[0], nc_edges[-1], 0, 1]) 
ax.set_xlabel("neighbourhood count") 
ax.set_ylabel("agent density")
ax.set_title("Final morans I heatmap over the neighbourhood count \n and agent density")
fig.colorbar(im, ax=ax, label="morans I")
plt.savefig(f"{OUTPUT_FOLDER}/sa_heatmap_morans.png", dpi=150)
plt.close()

plt.scatter(df.neighbourhood_count, df.morans_I_mean, c=df.density, s=6)
plt.xlabel("neighbourhood count")
plt.ylabel("morans I")
plt.title("Final morans I scatter plot over the neighbourhood count")
plt.savefig(f"{OUTPUT_FOLDER}/sa_scatter_morans.png", dpi=150)




df["x_bin"] = pd.cut(df["density"], edges)
df["y_bin"] = pd.cut(df["activation_rate"],         edges)

g = df.groupby(["y_bin", "x_bin"], observed=False)["nb_variance_mean"]
mean_grid, count_grid = g.mean().unstack(), g.size().unstack()
masked = np.ma.masked_where(count_grid.values < 5, mean_grid.values)

fig, ax = plt.subplots(figsize=(6.5, 5))
cmap = plt.cm.viridis.copy()
cmap.set_bad("lightgray")
im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
ax.set_xlabel("agent density") 
ax.set_ylabel("activation rate")
ax.set_title("Final neighbourhood variance heatmap over the agent density \n and activation rate")
fig.colorbar(im, ax=ax, label="neighbourhood variance")
plt.savefig(f"{OUTPUT_FOLDER}/sa_heatmap_nb_variance.png", dpi=150)
plt.close()

plt.scatter(df.density, df.nb_variance_mean, c=df.density, s=6)
plt.xlabel("agent density")
plt.ylabel("morans I")
plt.title("Final neighbourhood variance scatter plot over the agent density")
plt.savefig(f"{OUTPUT_FOLDER}/sa_scatter_nb_variance.png", dpi=150)

def dist(col, label, fname):
    vals = df[col].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=40, color="tab:blue", alpha=0.8)
    ax.axvline(vals.mean(), color="tab:red", linestyle="--",
               label=f"mean = {vals.mean():.3f}")
    ax.set_xlabel(label)
    ax.set_ylabel("number of combinations")
    ax.set_title(f"Distribution of final {label} across all parameter combinations")
    ax.legend()
    plt.savefig(f"{OUTPUT_FOLDER}/{fname}", dpi=150)
    plt.close()
    print(f"saved {fname}  (mean={vals.mean():.3f}, min={vals.min():.3f}, max={vals.max():.3f})")

dist("morans_I_mean",    "Moran's I",              "sa_morans_dist.png")
dist("nb_variance_mean", "neighbourhood variance", "sa_nb_variance_dist.png")
dist("H_tail_mean", "H segregation", "sa_H_dist.png")