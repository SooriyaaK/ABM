import os

import solara

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


path = os.path.dirname(os.path.abspath(__file__))


def agent_portrayal(agent):
    #base initial visualization of the agents
    style = AgentPortrayalStyle(
        x=agent.cell.coordinate[0],
        y=agent.cell.coordinate[1],
        marker="o",
        color="tab:gray",
        size=75,
    )
    #colours of agents when they are happy
    colors_happy = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    style.update(("color", colors_happy[agent.type]))
    if not agent.happy:
        colors_unhappy = {1: "lightblue", 2: "moccasin", 3: "lightgreen"}
        style.update(("color", colors_unhappy[agent.type]), ("size", 50), ("zorder", 2),)
    return style


model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "density": Slider("Agent density", 0.8, 0.1, 1.0, 0.1),
    "frac1": Slider("Type 1 percentage", 0.33, 0.0, 1.0, 0.05),
    "frac2": Slider("Type 2 percentage", 0.33, 0.0, 1.0, 0.05),
    "homophily": Slider("Homophily", 0.4, 0.0, 1.0, 0.125),
    "width": 20,
    "height": 20,
}

# Note: Models with images as markers are very performance intensive.
model1 = Schelling(scenario=SchellingScenario())
renderer = SpaceRenderer(model1, backend="matplotlib").setup_agents(agent_portrayal)
# Here we use renderer.render() to render the agents and grid in one go.
# This function always renders the grid and then renders the agents or
# property layers on top of it if specified.
renderer.render()

HappyPlot = make_plot_component({"happy": "tab:green"})

page = SolaraViz(
    model1,
    renderer,
    components=[
        HappyPlot,
        get_happy_agents,
    ],
    model_params=model_params,
)
page  # noqa
