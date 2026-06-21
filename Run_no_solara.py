import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Model import Schelling, SchellingScenario


def run_single(seed: int, max_steps: int, homophily: float,
               defector_frac: float, learning_rate: float) -> dict:
    """Run one simulation with given params and seed."""
    scenario = SchellingScenario(
        width=50,
        height=50,
        density=0.8,
        frac1=0.33,
        frac2=0.33,
        homophily=homophily,
        neighbourhood_count=10,
        defector_frac=defector_frac,
        learning_rate=learning_rate, 
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
    }


def save_combo_results(results: list[dict], combo_idx: int, params: dict,
                       job_id: str, max_steps: int):
    """Aggregate N seed results for one combo and save plot + npz."""
    os.makedirs("output", exist_ok=True)

    tag = (f"job{job_id}_combo{combo_idx}"
           f"_h{params['homophily']}"
           f"_d{params['defector_frac']}"
           f"_lr{params['learning_rate']}")

    # Stack arrays: shape (N, max_steps) or (N, max_steps+1)
    all_H       = np.array([r["H_series"]       for r in results])
    all_med     = np.array([r["median_utility"]  for r in results])
    all_q25     = np.array([r["q25_utility"]     for r in results])
    all_q75     = np.array([r["q75_utility"]     for r in results])
    all_steps   = np.array([r["steps_taken"]     for r in results])

    # Save raw data for aggregation later
    np.savez(
        f"output/run_{tag}.npz",
        all_H=all_H,
        all_median_utility=all_med,
        all_q25_utility=all_q25,
        all_q75_utility=all_q75,
        convergence_steps=all_steps,
        params=np.array([params["homophily"],
                         params["defector_frac"],
                         params["learning_rate"]]),
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
        f"homophily={params['homophily']}  defector_frac={params['defector_frac']}  "
        f"learning_rate={params['learning_rate']}  (N={N} seeds)",
        fontsize=10,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo-idx", type=int, default=0,
                        help="Index into params.json (= SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--n-seeds",   type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
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
            homophily=params["homophily"],
            defector_frac=params["defector_frac"],
            learning_rate=params["learning_rate"],
        )
        results.append(result)

    save_combo_results(results, args.combo_idx, params, job_id, args.max_steps)
