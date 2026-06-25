import numpy as np
import matplotlib.pyplot as plt
from Model import Schelling, SchellingScenario


MAX_STEPS = 500
defector_fracs = [0.1, 0.3, 0.5, 0.7]

for d in defector_fracs:
    scenario = SchellingScenario(
        seed=69,
        density=0.8,
        neighbourhood_count=25,
        activation_rate=0.3,
        defector_frac=d,
        max_steps=MAX_STEPS,
    )

    model = Schelling(scenario=scenario)
    series = []

    while model.running:

        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        series.append(
            df["defector_proportion"].iloc[-1]
        )

    plt.plot(
        range(len(series)),
        series,
        linewidth=2,
        label=f"Initial defectors = {d:.1f}"
    )


plt.xlabel("Step")
plt.ylabel("Defector proportion")
plt.title("Evolution of defection")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig("results/defector_comparison.png", dpi=150)
plt.show()