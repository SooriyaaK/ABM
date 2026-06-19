import os

import numpy as np
import solara
import matplotlib.pyplot as plt
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from matplotlib.collections import LineCollection

from Model import Schelling, SchellingScenario
from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle


def get_happy_agents(model):
    """Display a text count of how many happy agents there are."""
    return solara.Markdown(f"**Happy agents: {model.happy}**")

def get_segregation_status(model):
    """Display current H value, segregation level, and convergence status."""

    # H value
    H = model.H_history[-1] if model.H_history else None

    # Segregation level
    if H is None:
        level = "N/A"
        color = "gray"
    elif H >= 0.8:
        level = "*** High segregation"
        color = "red"
    elif H >= 0.4:
        level = "** Moderate segregation"
        color = "orange"
    else:
        level = "* Low segregation"
        color = "green"

    h_str = f"{H:.4f}" if H is not None else "N/A"

    # Convergence status
    if not model.running:
        all_happy = model.happy >= len(model.agents)

        segregation_converged = (
            len(model.H_history) >= model.convergence_window
            and (
                max(model.H_history[-model.convergence_window:])
                - min(model.H_history[-model.convergence_window:])
            ) < model.epsilon
        )

        if all_happy and segregation_converged:
            convergence_str = (
                f"**Converged** : all agents happy "
                f"and H stable within ε={model.epsilon} "
                f"for {model.convergence_window} steps."
            )
        elif all_happy:
            convergence_str = "**Converged** : all agents are happy."
        elif segregation_converged:
            convergence_str = (
                f"**Converged** : H stable within ε={model.epsilon} "
                f"for {model.convergence_window} steps."
            )
        else:
            convergence_str = "**Stopped** : maximum iterations reached."
    else:
        convergence_str = "**Running...**"

    return solara.Markdown(
        f"**Segregation index H:** `{h_str}` — <span style='color:{color}'>{level}</span>\n\n"
        f"{convergence_str}"
    )

def UtilityPlot(model):
    """Custom Solara component: median utility with Q25/Q75 IQR band over time."""
    df = model.datacollector.get_agent_vars_dataframe()

    if df.empty or "agent_type" not in df.columns:
        return solara.Markdown("*No utility data yet.*")

    # get per-step utility from agent datacollector
    # agent_vars_dataframe has MultiIndex (Step, AgentID)
    utility_df = model.datacollector.get_agent_vars_dataframe()

    # Mesa stores agent vars; we need current_utility — add it to datacollector (see note below)
    # For now, use the model-level utility_history if available
    if not hasattr(model, "utility_history") or len(model.utility_history) == 0:
        return solara.Markdown("*No utility history yet.*")

    steps = np.arange(len(model.utility_history))
    medians = np.array([u["median"] for u in model.utility_history])
    q25s    = np.array([u["q25"]    for u in model.utility_history])
    q75s    = np.array([u["q75"]    for u in model.utility_history])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(steps, medians, color="tab:blue", label="Median utility")
    ax.fill_between(steps, q25s, q75s, color="tab:blue", alpha=0.2, label="IQR (Q25–Q75)")
    ax.set_title("Agent utility over time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Utility")
    ax.legend(fontsize=8)
    plt.tight_layout()

    solara.FigureMatplotlib(fig)
    plt.close(fig)


path = os.path.dirname(os.path.abspath(__file__))

def neighbourhood_portrayal(layer):
    return PropertyLayerStyle(colormap="tab20", alpha=0.4, colorbar=False)

def neighbourhood_borders(ax):
    cton = model1.cell_to_neighbourhood
    segments = []
    for (x, y), nb in cton.items():
        right_nb = cton.get((x + 1, y))
        if right_nb == None or right_nb.id != nb.id:
            segments.append([(x + 0.5, y - 0.5), (x + 0.5, y + 0.5)])
        top_nb = cton.get((x, y + 1))
        if top_nb == None or top_nb.id != nb.id:
            segments.append([(x - 0.5, y + 0.5), (x + 0.5, y + 0.5)])
    ax.add_collection(LineCollection(segments, colors="black", linewidths=1.5))

def agent_portrayal(agent):
    #base initial visualization of the agents
    style = AgentPortrayalStyle(
        x=agent.cell.coordinate[0],
        y=agent.cell.coordinate[1],
        marker="o",
        color="tab:gray",
        edgecolors = "black",
        linewidths = 0,
        size=75,
    )
    #colours of agents when they are happy
    colors_happy = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    if agent.strategy == "D":
        style.update(("linewidths", 1))
    style.update(("color", colors_happy[agent.type]))
    if not agent.happy:
        colors_unhappy = {1: "lightblue", 2: "moccasin", 3: "lightgreen"}
        style.update(("color", colors_unhappy[agent.type]), ("size", 50), ("zorder", 2),)
    return style


model_params = {
    "rng": {
        "type": "InputText",
        "value": 69,
        "label": "Random Seed",
    },
    "density": Slider("Agent density", 0.8, 0.1, 1.0, 0.1),
    "frac1": Slider("Type 1 percentage", 0.33, 0.0, 1.0, 0.05),
    "frac2": Slider("Type 2 percentage", 0.33, 0.0, 1.0, 0.05),
    "homophily": Slider("Homophily", 0.4, 0.0, 1.0, 0.125),
    "width": 50,
    "height": 50,
    "defector_frac": Slider("Defector fraction", 0.1, 0.0, 1.0, 0.05),

}

# Note: Models with images as markers are very performance intensive.
model1 = Schelling(scenario=SchellingScenario())
renderer = SpaceRenderer(model1, backend="matplotlib").setup_agents(agent_portrayal)
renderer.post_process = neighbourhood_borders

class _RerunRenderer(type(renderer)):
    @property
    def _post_process_applied(self):
        return False
    @_post_process_applied.setter
    def _post_process_applied(self, value):
        pass

renderer.__dict__.pop("_post_process_applied", None)
renderer.__class__ = _RerunRenderer
# Here we use renderer.render() to render the agents and grid in one go.
# This function always renders the grid and then renders the agents or
# property layers on top of it if specified.
renderer.render()

# HappyPlot = make_plot_component({"happy": "tab:green"})

HPlot = make_plot_component({"H": "tab:red"}) # plots segregation metric

page = SolaraViz(
    model1,
    renderer,
    components=[
        UtilityPlot,
        HPlot,
        get_happy_agents,
        get_segregation_status,
    ],
    model_params=model_params,
)

page  # noqa