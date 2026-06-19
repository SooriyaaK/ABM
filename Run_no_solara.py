import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Model import Schelling, SchellingScenario


def run_single(seed: int, max_steps: int = 200):
    """Run one simulation with a fixed seed, return collected data."""
    scenario = SchellingScenario(
        width=50,
        height=50,
        density=0.8,
        frac1=0.33,
        frac2=0.33,
        homophily=0.4,
        neighbourhood_count=10,
    )
    model = Schelling(scenario=scenario, rng=seed)

    median_utility, q25_utility, q75_utility = [], [], []

    # step 0
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

    # Pad to max_steps
    H_series      += [H_series[-1]]      * (max_steps - len(H_series))
    pad = max_steps + 1
    median_utility += [median_utility[-1]] * (pad - len(median_utility))
    q25_utility    += [q25_utility[-1]]    * (pad - len(q25_utility))
    q75_utility    += [q75_utility[-1]]    * (pad - len(q75_utility))

    final_H = model.H_history[-1] if model.H_history else None
    print(f"[seed={seed}] steps={steps_taken}, final H={final_H:.4f}, "
          f"happy={model.happy}/{len(model.agents)}")

    return {
        "seed": seed,
        "steps_taken": steps_taken,
        "H_series": np.array(H_series),
        "median_utility": np.array(median_utility),
        "q25_utility": np.array(q25_utility),
        "q75_utility": np.array(q75_utility),
    }


def save_results(result: dict, job_id: str, neighbourhood_count: int):
    """Save per-seed plot and numpy data to output/."""
    os.makedirs("output", exist_ok=True)
    seed = result["seed"]

    # Save raw arrays for later aggregation
    np.savez(
        f"output/run_job{job_id}_seed{seed}_nb{neighbourhood_count}.npz",
        **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
        seed=seed,
        steps_taken=result["steps_taken"],
    )

    # Save per-seed plot
    max_steps = len(result["H_series"])
    steps_H = np.arange(max_steps)
    steps_u = np.arange(max_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(steps_H, result["H_series"], color="tab:red")
    axes[0].axhline(0.8, color="red",    linestyle="--", linewidth=0.8, label="High (0.8)")
    axes[0].axhline(0.4, color="orange", linestyle="--", linewidth=0.8, label="Moderate (0.4)")
    axes[0].set_title(f"Segregation H — seed {seed}")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("H")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].plot(steps_u, result["median_utility"], color="tab:blue", label="Median utility")
    axes[1].fill_between(steps_u, result["q25_utility"], result["q75_utility"],
                         color="tab:blue", alpha=0.2, label="IQR (Q25–Q75)")
    axes[1].set_title(f"Agent utility — seed {seed}")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Utility")
    axes[1].legend()

    plt.tight_layout()
    fname = f"output/plot_job{job_id}_seed{seed}_nb{neighbourhood_count}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed / SLURM array task ID")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    job_id  = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
    nb_count = 10  # neighbourhood_count — keep in sync with scenario above

    result = run_single(seed=args.seed, max_steps=args.max_steps)
    save_results(result, job_id=job_id, neighbourhood_count=nb_count)
