#File that analyzes the npz files on the supercomputer and creates plots related to the convergence measure
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(42)

OUTPUT_FOLDER = "results_saltelli"

def analyse_npz(npz_glob: str, results_dir: str, tail: int = 50):
    os.makedirs(results_dir, exist_ok=True)
    files = sorted(glob.glob(npz_glob))
    if not files:
        print(f"\n(no npz files matched {npz_glob!r} — run the sweep first)")
        return
    print(f"\n=== NPZ analysis: {len(files)} files ({npz_glob}) ===")

    #convergence dynamics: H over time for a few combos
    # a random sample of combos to plot over
    sample = random.sample(files, min(8, len(files)))
    WINDOW = 10
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, f in enumerate(sample):                                
        d = np.load(f)
        mean_H = d["all_H"].mean(axis=0)                  # mean over seeds
        smooth = np.convolve(mean_H, np.ones(WINDOW) / WINDOW, mode="valid")
        x = np.arange(WINDOW - 1, len(mean_H))
        ax.plot(mean_H, alpha=0.2, linewidth=1, label="pre-smoothing" if j == 0 else None)
        ax.plot(x, smooth, alpha=0.8, linewidth=1.2, label="smoothed (window 10)" if j == 0 else None)
    ax.set_xlabel("Step"); ax.set_ylabel("H (mean over seeds)")
    ax.set_title("Segregation trajectories (sample of combos)")
    ax.legend()
    plt.tight_layout(); plt.savefig(f"{results_dir}/sa_H_trajectories.png", dpi=150); plt.close()
    print(f"  saved {results_dir}/sa_H_trajectories.png")

    #mean H trajectory across all runs
    all_curves = np.array([np.load(f)["all_H"].mean(axis=0) for f in files])
    m, s = all_curves.mean(axis=0), all_curves.std(axis=0)
    steps = np.arange(len(m))
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(steps, m, color="tab:red", label="mean over all combos")
    ax2.fill_between(steps, m - s, m + s, alpha=0.2, color="tab:red", label="±1 std across combos")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("H (mean over seeds)")
    ax2.set_title("Mean segregation trajectory over all parameter combinations")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/sa_H_trajectories_all.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-glob", default=f"{OUTPUT_FOLDER}/run_*.npz")
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--results-dir", default="sa_results",
                    help="Folder to save the H trajectory plot into")
    args = ap.parse_args()

    analyse_npz(args.npz_glob, args.results_dir, args.tail)
