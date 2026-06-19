import numpy as np
import matplotlib.pyplot as plt
from Model import Schelling, SchellingScenario


def run_multiple_time(N=10):
    """
    Runs simulation N times, generates plots with std as error bars for H and happiness.
    """

    # config
    MAX_STEPS = 200

    # history
    all_H = []
    all_happy = []
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

        # step through simulation
        for step in range(MAX_STEPS):
            if not model.running:
                steps_taken = step
                print(f"  Stopped at step {step}.")
                break
            model.step()
        else:
            steps_taken = step
            print(f"  Reached max steps (step={step}).")

        convergence_steps.append(steps_taken)

        # retrieve collected data
        df = model.datacollector.get_model_vars_dataframe()
        final_H = model.H_history[-1] if model.H_history else None
        print(f"-> Final H: {final_H:.4f}" if final_H is not None else "  No H recorded.")
        print(f"-> Happy agents: {model.happy} / {len(model.agents)}")

        # Pad shorter runs with their last value so all runs are the same length
        # this is important for plotting.
        H_series = model.H_history[1:]
        happy_series = list(df["happy"])[1:]
        H_series += [H_series[-1]] * (MAX_STEPS - len(H_series))
        happy_series += [happy_series[-1]] * (MAX_STEPS - len(happy_series))

        all_H.append(H_series)
        all_happy.append(happy_series)

    # Convert to arrays: shape (N, MAX_STEPS)
    all_H = np.array(all_H)
    all_happy = np.array(all_happy)

    # stats
    mean_H = all_H.mean(axis=0)
    std_H = all_H.std(axis=0)
    mean_happy = all_happy.mean(axis=0)
    std_happy = all_happy.std(axis=0)

    steps = np.arange(MAX_STEPS)
    convergence_steps = np.array(convergence_steps)
    n_maxed = (convergence_steps == MAX_STEPS).sum()

    # Init plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # H over time
    axes[0].plot(steps, mean_H, color="tab:red", label="Mean H")
    axes[0].fill_between(steps, mean_H - std_H, mean_H + std_H,
                         color="tab:red", alpha=0.2, label="±1 std")
    axes[0].axhline(0.8, color="red",    linestyle="--", linewidth=0.8, label="High (0.8)")
    axes[0].axhline(0.4, color="orange", linestyle="--", linewidth=0.8, label="Moderate (0.4)")
    axes[0].set_title(f"Segregation index H over time (N={N})")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("H")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Happy agents over time
    axes[1].plot(steps, mean_happy, color="tab:green", label="Mean happy")
    axes[1].fill_between(steps, mean_happy - std_happy, mean_happy + std_happy,
                         color="tab:green", alpha=0.2, label="±1 std")
    axes[1].set_title(f"Happy agents over time (N={N})")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Happy agents")
    axes[1].legend()

    # Convergence histogram; counts per H
    bins = np.arange(convergence_steps.min(), convergence_steps.max() + 2) - 0.5 # nice bins per integer
    axes[2].hist(convergence_steps, bins=bins, color="tab:pink", edgecolor="white")
    if n_maxed > 0:
        axes[2].axvline(MAX_STEPS, color="red", linestyle="--", linewidth=1,
                        label=f"Max steps hit ({n_maxed}x)")
        axes[2].legend()
    axes[2].set_title(f"Steps until convergence (N={N})")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Count")

    # Summarise stats below histogram
    axes[2].text(0.05, 0.95,
                 f"Mean: {convergence_steps.mean():.1f}\n"
                 f"Std:  {convergence_steps.std():.1f}\n"
                 f"Min:  {convergence_steps.min()}\n"
                 f"Max:  {convergence_steps.max()}",
                 transform=axes[2].transAxes,
                 verticalalignment="top",
                 fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    print("steps:", steps.shape)
    print("mean_H:", mean_H.shape)
    print("mean_happy:", mean_happy.shape)

    # Render plots
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()
    print("\nPlot saved to results.png")


if __name__ == "__main__":
    run_multiple_time(N=2)
