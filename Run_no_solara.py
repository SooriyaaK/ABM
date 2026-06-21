import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Model import Schelling, SchellingScenario
from Clustering import compute_morans_I, compute_neighbourhood_income_variance
from scipy.stats import gaussian_kde


def run_single(seed: int, max_steps: int, density: float,
               defector_frac: float, neighbourhood_count: int,
               activation_rate: float) -> dict:
    """Run one simulation with given params and seed."""
    scenario = SchellingScenario(
        width=50,
        height=50,
        density=density,
        frac1=0.33,
        frac2=0.33,
        homophily=0.4,
        neighbourhood_count=neighbourhood_count,
        defector_frac=defector_frac,
        activation_rate = activation_rate,
        seed=seed, 
    )
    model = Schelling(scenario=scenario)

    median_utility, q25_utility, q75_utility = [], [], []

    utils = [a.current_utility for a in model.agents]
    median_utility.append(np.median(utils))
    q25_utility.append(np.percentile(utils, 25))
    q75_utility.append(np.percentile(utils, 75))

    steps_taken = max_steps
    for step in range(max_steps):
        if not model.running:
            steps_taken = step
            break
        model.step()
        utils = [a.current_utility for a in model.agents]
        median_utility.append(np.median(utils))
        q25_utility.append(np.percentile(utils, 25))
        q75_utility.append(np.percentile(utils, 75))

    H_series = model.H_history[1:]
    H_series       += [H_series[-1]]       * (max_steps - len(H_series))
    pad = max_steps + 1
    median_utility += [median_utility[-1]] * (pad - len(median_utility))
    q25_utility    += [q25_utility[-1]]    * (pad - len(q25_utility))
    q75_utility    += [q75_utility[-1]]    * (pad - len(q75_utility))

    final_H = model.H_history[-1] if model.H_history else None
    print(f"  [seed={seed}] steps={steps_taken}, H={final_H:.4f}, "
          f"happy={model.happy}/{len(model.agents)}")

    return {
        "seed": seed,
        "steps_taken": steps_taken,
        "H_series":        np.array(H_series),
        "median_utility":  np.array(median_utility),
        "q25_utility":     np.array(q25_utility),
        "q75_utility":     np.array(q75_utility),
        "final_morans_I":  compute_morans_I(model),           # single final value
        "final_nb_variance": compute_neighbourhood_income_variance(model),
    }


def save_combo_results(results: list[dict], combo_idx: int, params: dict,
                       job_id: str, max_steps: int):
    """Aggregate N seed results for one combo and save plot + npz."""
    os.makedirs("output", exist_ok=True)

    tag = (f"job{job_id}_combo{combo_idx}"
       f"_den{params['density']}"
       f"_df{params['defector_frac']}"
       f"_nc{params['neighbourhood_count']}"
       f"_ar{params['activation_rate']}")

    # Stack arrays: shape (N, max_steps) or (N, max_steps+1)
    all_H       = np.array([r["H_series"]       for r in results])
    all_med     = np.array([r["median_utility"]  for r in results])
    all_q25     = np.array([r["q25_utility"]     for r in results])
    all_q75     = np.array([r["q75_utility"]     for r in results])
    all_steps   = np.array([r["steps_taken"]     for r in results])
    all_morans   = np.array([r["final_morans_I"]    for r in results])
    all_nb_var   = np.array([r["final_nb_variance"] for r in results])

    # Save raw data for aggregation later
    np.savez(
        f"output/run_{tag}.npz",
        all_H=all_H,
        all_median_utility=all_med,
        all_q25_utility=all_q25,
        all_q75_utility=all_q75,
        convergence_steps=all_steps,
        final_morans_I=all_morans,        
        final_nb_variance=all_nb_var,
        params=np.array([params["density"],
                params["defector_frac"],
                params["neighbourhood_count"],
                params["activation_rate"]]),
            )

    # Summary plot for this combo
    mean_H  = all_H.mean(axis=0)
    std_H   = all_H.std(axis=0)
    grand_median = np.median(all_med, axis=0)
    grand_q25    = all_q25.mean(axis=0)
    grand_q75    = all_q75.mean(axis=0)

    steps_H = np.arange(max_steps)
    steps_u = np.arange(max_steps + 1)
    n_maxed = (all_steps == max_steps).sum()
    N = len(results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(
        f"density={params['density']}  defector_frac={params['defector_frac']}  "
        f"neighbourhood_count={params['neighbourhood_count']}  activation_rate={params['activation_rate']} "
        f"(N={N} seeds)", fontsize=10,
    )

    # H over time
    axes[0].plot(steps_H, mean_H, color="tab:red", label="Mean H")
    axes[0].fill_between(steps_H, mean_H - std_H, mean_H + std_H,
                         color="tab:red", alpha=0.2, label="±1 std")
    axes[0].axhline(0.8, color="red",    linestyle="--", linewidth=0.8, label="High (0.8)")
    axes[0].axhline(0.4, color="orange", linestyle="--", linewidth=0.8, label="Moderate (0.4)")
    axes[0].set_title("Segregation index H")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("H")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Utility over time
    axes[1].plot(steps_u, grand_median, color="tab:blue", label="Median utility")
    axes[1].fill_between(steps_u, grand_q25, grand_q75,
                         color="tab:blue", alpha=0.2, label="IQR (Q25–Q75)")
    axes[1].set_title("Agent utility")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Utility")
    axes[1].legend()

    # Convergence histogram
    bins = np.arange(all_steps.min(), all_steps.max() + 2) - 0.5
    axes[2].hist(all_steps, bins=bins, color="tab:pink", edgecolor="white")
    if n_maxed > 0:
        axes[2].axvline(max_steps, color="red", linestyle="--", linewidth=1,
                        label=f"Max steps hit ({n_maxed}x)")
        axes[2].legend()
    axes[2].set_title("Steps until convergence")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Count")
    axes[2].text(0.05, 0.95,
                 f"Mean: {all_steps.mean():.1f}\nStd: {all_steps.std():.1f}\n"
                 f"Min: {all_steps.min()}\nMax: {all_steps.max()}",
                 transform=axes[2].transAxes, verticalalignment="top",
                 fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    fname = f"output/plot_{tag}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle(
        f"Clustering metrics — homophily={params['homophily']}  "
        f"defector_frac={params['defector_frac']}  learning_rate={params['learning_rate']}  "
        f"(N={N} seeds)",
        fontsize=10,
    )

    # Moran's I distribution across seeds
    ax = axes2[0]
    if len(all_morans) > 1 and all_morans.std() > 0:
        kde = gaussian_kde(all_morans)
        x = np.linspace(-1, 1, 200)
        ax.plot(x, kde(x), color="tab:green", linewidth=2)
        ax.fill_between(x, kde(x), alpha=0.3, color="tab:green")
    ax.scatter(all_morans, np.zeros_like(all_morans) - 0.02,
            color="tab:green", s=40, zorder=5, label="Seeds")
    ax.axvline(all_morans.mean(), color="black", linestyle="--",
            linewidth=1, label=f"Mean: {all_morans.mean():.3f}")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Moran's I (income clustering)")
    ax.set_xlabel("Moran's I")
    ax.set_xlim(-1, 1)
    ax.legend()
    ax.text(0.05, 0.95,
            f"Mean: {all_morans.mean():.3f}\nStd: {all_morans.std():.3f}\n"
            f"Min: {all_morans.min():.3f}\nMax: {all_morans.max():.3f}",
            transform=ax.transAxes, verticalalignment="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Neighbourhood income variance
    ax = axes2[1]
    if len(all_nb_var) > 1 and all_nb_var.std() > 0:
        kde = gaussian_kde(all_nb_var)
        x = np.linspace(0, 1, 200)
        ax.plot(x, kde(x), color="tab:purple", linewidth=2)
        ax.fill_between(x, kde(x), alpha=0.3, color="tab:purple")
    ax.scatter(all_nb_var, np.zeros_like(all_nb_var) - 0.02,
            color="tab:purple", s=40, zorder=5, label="Seeds")
    ax.axvline(all_nb_var.mean(), color="black", linestyle="--",
            linewidth=1, label=f"Mean: {all_nb_var.mean():.3f}")
    ax.set_title("Between-neighbourhood income variance (normalized)")
    ax.set_xlabel("Variance (0=mixed, 1=segregated)")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.text(0.05, 0.95,
            f"Mean: {all_nb_var.mean():.3f}\nStd: {all_nb_var.std():.3f}\n"
            f"Min: {all_nb_var.min():.3f}\nMax: {all_nb_var.max():.3f}",
            transform=ax.transAxes, verticalalignment="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    fname2 = f"output/clustering_{tag}.png"
    plt.savefig(fname2, dpi=150)
    plt.close()
    print(f"Saved: {fname2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo-idx", type=int, default=0,
                        help="Index into params.json (= SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--n-seeds",   type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--params-file", type=str, default="params.json")
    args = parser.parse_args()

    # Load parameter combo for this task
    with open(args.params_file) as f:
        all_combos = json.load(f)
    params = all_combos[args.combo_idx]

    print(f"\n=== Combo {args.combo_idx}/{len(all_combos)-1}: {params} ===")

    job_id = os.environ.get("SLURM_ARRAY_JOB_ID",
             os.environ.get("SLURM_JOB_ID", "local"))

    results = []
    for seed in range(args.n_seeds):
        result = run_single(
        seed=seed,
        max_steps=args.max_steps,
        density=params["density"],
        defector_frac=params["defector_frac"],
        neighbourhood_count=params["neighbourhood_count"],
        activation_rate=params["activation_rate"],
        )
        results.append(result)

    save_combo_results(results, args.combo_idx, params, job_id, args.max_steps)
