import numpy as np
import matplotlib.pyplot as plt
from Model import Schelling, SchellingScenario
import os


def run_multiple_time(N=10):
    """
    Runs simulation N times, generates plots with std/IQR as error bands for H and utility.
    """

    # config
    MAX_STEPS = 200

    # history
    all_H = []
    all_median_utility = []
    all_q25_utility = []
    all_q75_utility = []
    convergence_steps = []

    # run simulation N times
    for i in range(N):
        print(f"\n--- Run {i + 1}/{N} ---")

        scenario = SchellingScenario(
            width=50,
            height=50,
            density=0.8,
            frac1=0.33,
            frac2=0.33,
            homophily=0.4,
            neighbourhood_count=10,
        )

        # set up model
        model = Schelling(scenario=scenario)

        # per-step utility tracking for this run
        run_median_utility = []
        run_q25_utility = []
        run_q75_utility = []

        # collect initial utility (step 0)
        utilities_t0 = [a.current_utility for a in model.agents]
        run_median_utility.append(np.median(utilities_t0))
        run_q25_utility.append(np.percentile(utilities_t0, 25))
        run_q75_utility.append(np.percentile(utilities_t0, 75))

        # step through simulation
        steps_taken = MAX_STEPS
        for step in range(MAX_STEPS):
            if not model.running:
                steps_taken = step
                print(f"  Stopped at step {step}.")
                break
            model.step()

            # collect utility after each step
            utilities = [a.current_utility for a in model.agents]
            run_median_utility.append(np.median(utilities))
            run_q25_utility.append(np.percentile(utilities, 25))
            run_q75_utility.append(np.percentile(utilities, 75))
        else:
            print(f"  Reached max steps ({MAX_STEPS}).")

        convergence_steps.append(steps_taken)

        final_H = model.H_history[-1] if model.H_history else None
        print(f"-> Final H: {final_H:.4f}" if final_H is not None else "  No H recorded.")
        print(f"-> Happy agents: {model.happy} / {len(model.agents)}")

        # Pad shorter runs with their last value
        H_series = model.H_history[1:]
        H_series += [H_series[-1]] * (MAX_STEPS - len(H_series))

        pad_len = MAX_STEPS + 1  # +1 for step 0
        run_median_utility += [run_median_utility[-1]] * (pad_len - len(run_median_utility))
        run_q25_utility    += [run_q25_utility[-1]]    * (pad_len - len(run_q25_utility))
        run_q75_utility    += [run_q75_utility[-1]]    * (pad_len - len(run_q75_utility))

        all_H.append(H_series)
        all_median_utility.append(run_median_utility)
        all_q25_utility.append(run_q25_utility)
        all_q75_utility.append(run_q75_utility)

    # Convert to arrays
    all_H             = np.array(all_H)              # (N, MAX_STEPS)
    all_median_utility = np.array(all_median_utility) # (N, MAX_STEPS+1)
    all_q25_utility    = np.array(all_q25_utility)
    all_q75_utility    = np.array(all_q75_utility)

    # Stats across runs
    mean_H   = all_H.mean(axis=0)
    std_H    = all_H.std(axis=0)

    # Median of medians across runs, and mean of IQR bounds
    grand_median_utility = np.median(all_median_utility, axis=0)
    grand_q25            = all_q25_utility.mean(axis=0)
    grand_q75            = all_q75_utility.mean(axis=0)

    steps_H = np.arange(MAX_STEPS)
    steps_u = np.arange(MAX_STEPS + 1)
    convergence_steps = np.array(convergence_steps)
    n_maxed = (convergence_steps == MAX_STEPS).sum()

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # H over time
    axes[0].plot(steps_H, mean_H, color="tab:red", label="Mean H")
    axes[0].fill_between(steps_H, mean_H - std_H, mean_H + std_H,
                         color="tab:red", alpha=0.2, label="±1 std")
    axes[0].axhline(0.8, color="red",    linestyle="--", linewidth=0.8, label="High (0.8)")
    axes[0].axhline(0.4, color="orange", linestyle="--", linewidth=0.8, label="Moderate (0.4)")
    axes[0].set_title(f"Segregation index H over time (N={N})")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("H")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Median utility over time 
    axes[1].plot(steps_u, grand_median_utility, color="tab:blue", label="Median utility")
    axes[1].fill_between(steps_u, grand_q25, grand_q75,
                         color="tab:blue", alpha=0.2, label="IQR (Q25–Q75)")
    axes[1].set_title(f"Agent utility over time (N={N})")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Utility")
    axes[1].legend()

    # Convergence histogram
    bins = np.arange(convergence_steps.min(), convergence_steps.max() + 2) - 0.5
    axes[2].hist(convergence_steps, bins=bins, color="tab:pink", edgecolor="white")
    if n_maxed > 0:
        axes[2].axvline(MAX_STEPS, color="red", linestyle="--", linewidth=1,
                        label=f"Max steps hit ({n_maxed}x)")
        axes[2].legend()
    axes[2].set_title(f"Steps until convergence (N={N})")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Count")
    axes[2].text(0.05, 0.95,
                 f"Mean: {convergence_steps.mean():.1f}\n"
                 f"Std:  {convergence_steps.std():.1f}\n"
                 f"Min:  {convergence_steps.min()}\n"
                 f"Max:  {convergence_steps.max()}",
                 transform=axes[2].transAxes,
                 verticalalignment="top",
                 fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Build filename from SLURM job ID + key params
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    fname = f"results/results_job{job_id}_N{N}_nb{scenario.neighbourhood_count}.png"

    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"\nPlot saved to {fname}")


if __name__ == "__main__":
    run_multiple_time(N=2)
