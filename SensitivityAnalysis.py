"""
Sensitivity analysis for the Schelling gentrification sweep.

Your params.json is a FULL FACTORIAL GRID (every density × defector_frac ×
neighbourhood_count × activation_rate), NOT a Saltelli/Morris sample — so
SALib.analyze.sobol does not apply directly. But on a complete grid you can
compute first-order Sobol indices exactly via functional ANOVA (averaging the
output over all other parameters at each level of one parameter).

Two parts:
  A) on results_summary.csv  -> the sensitivity analysis (indices + main effects)
  B) on the output/*.npz     -> seed-noise check + convergence dynamics

Usage:
    python SensitivityAnalysis.py                       # uses results_summary.csv + output/
    python SensitivityAnalysis.py --csv results_test.csv --npz-glob "output_test/run_*.npz"
"""
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAMS = ["density", "defector_frac", "neighbourhood_count", "activation_rate"]
# the segregation outputs available in the summary CSV
OUTPUTS = ["H_tail_mean", "morans_I_mean", "nb_variance_mean"]


# ----------------------------------------------------------------------------
# A) SENSITIVITY ON THE SUMMARY CSV
# ----------------------------------------------------------------------------
def first_order_indices(df: pd.DataFrame, output: str) -> dict:
    """
    First-order Sobol index for each parameter, computed exactly from a full grid:
        S_i = Var_x( E[Y | X_i = x] ) / Var(Y)
    i.e. how much of the output variance is explained by varying that parameter
    alone (averaging out all the others). Uses population variance (ddof=0).
    """
    total_var = df[output].var(ddof=0)
    out = {}
    for p in PARAMS:
        cond_means = df.groupby(p)[output].mean()       # E[Y | X_i = level]
        out[p] = float(cond_means.var(ddof=0) / total_var) if total_var > 0 else 0.0
    return out


def total_order_indices(df: pd.DataFrame, output: str) -> dict:
    """
    Total-order Sobol index for each parameter, exact on a full grid:
        S_Ti = 1 - Var( E[Y | X_{~i}] ) / Var(Y)
    Fix all parameters EXCEPT i (group by the others), average Y over i; the
    variance of those group-means is the part NOT due to i. One minus that is
    i's total effect = its own main effect plus every interaction it's in.
    The gap S_Ti - S_i tells you how much of i's influence is interaction.
    """
    total_var = df[output].var(ddof=0)
    out = {}
    for p in PARAMS:
        others = [q for q in PARAMS if q != p]
        cond_means = df.groupby(others)[output].mean()   # E[Y | X_{~i}], averaging over i
        out[p] = float(1.0 - cond_means.var(ddof=0) / total_var) if total_var > 0 else 0.0
    return out


def regression_sensitivity(df: pd.DataFrame, output: str) -> dict:
    """
    Cross-check: standardized linear-regression coefficients (|beta| on z-scored
    inputs). A quick, interpretable 'how strongly does each parameter push the
    output' — sign tells direction, magnitude tells strength.
    """
    X = df[PARAMS].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    X = X.values
    y = (df[output].values - df[output].mean()) / df[output].std(ddof=0)
    # least squares with intercept
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {p: float(beta[i + 1]) for i, p in enumerate(PARAMS)}


def plot_main_effects(df: pd.DataFrame, output: str, fname: str):
    """One panel per parameter: mean output vs that parameter's level."""
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(4 * len(PARAMS), 3.5), sharey=True)
    for ax, p in zip(axes, PARAMS):
        g = df.groupby(p)[output]
        m, s = g.mean(), g.std(ddof=0)
        ax.plot(m.index, m.values, "o-", color="tab:blue")
        ax.fill_between(m.index, m.values - s.values, m.values + s.values,
                        alpha=0.2, color="tab:blue")
        ax.set_xlabel(p)
        ax.set_title(p, fontsize=9)
    axes[0].set_ylabel(output)
    fig.suptitle(f"Main effects on {output}", fontsize=11)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  saved {fname}")


def analyse_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"\n=== CSV analysis: {csv_path}  ({len(df)} combos) ===")
    for output in OUTPUTS:
        if output not in df.columns:
            continue
        S = first_order_indices(df, output)
        ST = total_order_indices(df, output)
        reg = regression_sensitivity(df, output)
        print(f"\n  Output: {output}")
        print(f"  {'parameter':<22}{'first-order S_i':<17}{'total-order S_Ti':<18}"
              f"{'interaction (ST-S)':<20}{'reg.beta':<10}")
        for p in PARAMS:
            print(f"  {p:<22}{S[p]:<17.3f}{ST[p]:<18.3f}{ST[p]-S[p]:<20.3f}{reg[p]:<10.3f}")
        print(f"  {'sum of first-order':<22}{sum(S.values()):<17.3f}"
              f"(remainder {1 - sum(S.values()):.3f} = interactions + noise)")
        plot_main_effects(df, output, f"sa_main_effects_{output}.png")


# ----------------------------------------------------------------------------
# B) INSPECTION ON THE RAW NPZ FILES
# ----------------------------------------------------------------------------
def analyse_npz(npz_glob: str, tail: int = 50):
    files = sorted(glob.glob(npz_glob))
    if not files:
        print(f"\n(no npz files matched {npz_glob!r} — skipping raw analysis)")
        return
    print(f"\n=== NPZ analysis: {len(files)} files ({npz_glob}) ===")

    # 1. seed-to-seed noise: is the metric stable enough for SA?
    within_seed_std, across_combo_spread = [], []
    for f in files:
        d = np.load(f)
        tail_H = d["all_H"][:, -tail:].mean(axis=1)      # one value per seed
        within_seed_std.append(tail_H.std())             # noise within a combo
        across_combo_spread.append(tail_H.mean())        # this combo's value
    within = np.mean(within_seed_std)
    across = np.std(across_combo_spread)
    print(f"  mean within-combo seed std of H : {within:.4f}  (your stochastic noise)")
    print(f"  std of H across combos          : {across:.4f}  (parameter-driven signal)")
    print(f"  signal-to-noise (across/within) : {across / within:.1f}x"
          if within > 0 else "  (no within-seed variation)")
    print("  -> if signal >> noise, your SA indices are trustworthy.")

    # 2. convergence dynamics: H over time for a few combos
    fig, ax = plt.subplots(figsize=(7, 4))
    for f in files[:8]:                                   # first few combos
        d = np.load(f)
        mean_H = d["all_H"].mean(axis=0)                  # mean over seeds
        ax.plot(mean_H, alpha=0.8, linewidth=1)
    ax.set_xlabel("Step"); ax.set_ylabel("H (mean over seeds)")
    ax.set_title("Segregation trajectories (sample of combos)")
    plt.tight_layout(); plt.savefig("sa_H_trajectories.png", dpi=150); plt.close()
    print("  saved sa_H_trajectories.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_summary.csv")
    ap.add_argument("--npz-glob", default="output/run_*.npz")
    ap.add_argument("--tail", type=int, default=50)
    args = ap.parse_args()

    analyse_csv(args.csv)
    analyse_npz(args.npz_glob, args.tail)
