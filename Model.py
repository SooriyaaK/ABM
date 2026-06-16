from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from Agents import SchellingAgent
from mesa.experimental.scenarios import Scenario


class SchellingScenario(Scenario):
    """Scenario for the Schelling model.

    Args:
        width: Width of the grid
        height: Height of the grid
        density: Initial chance for a cell to be populated (0-1)
        minority_pc: Chance for an agent to be in minority class (0-1)
        homophily: Minimum number of similar neighbors needed for happiness
        radius: Search radius for checking neighbor similarity
        rng: Seed for reproducibility
    """

    height: int = 50
    width: int = 50
    density: float = 0.8
    frac1: float = 0.33
    frac2: float = 0.33
    frac3: float = 0.33
    homophily: float = 0.4
    radius: int = 1
    housing_cost: float = 10.0


class Schelling(Model):
    """Model class for the Schelling segregation model."""

    def __init__(self, scenario: SchellingScenario = SchellingScenario):
        """Create a new Schelling model.

        Args:
            scenario: SchellingScenario containing model parameters.
        """
        super().__init__(scenario=scenario)

        # Model parameters
        self.density = scenario.density
        self.frac1 = scenario.frac1
        self.frac2 = scenario.frac2
        self.frac3 = max(0.0, 1.0 - self.frac1 - self.frac2)

        # Initialize grid
        self.grid = OrthogonalMooreGrid(
            (scenario.width, 
             scenario.height), 
             random=self.random, 
             capacity=1,
        )

        for cell in self.grid.all_cells.cells:
            x, y = cell.coordinate

            housing_cost = 1000
            distance = x + y 
            cell.housing_cost = housing_cost + distance * 10


        # Track happiness
        self.happy = 0

        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "happy": "happy",
                "pct_happy": lambda m: (
                    (m.happy / len(m.agents)) * 100 if len(m.agents) > 0 else 0
                ),
                "population": lambda m: len(m.agents),
                "frac1": "frac1",
                "frac2": "frac2",
                "frac3": "frac3",
                #"minority_pct": lambda m: (
                #    sum(1 for agent in m.agents if agent.type == 1)
                #    / len(m.agents)
                #    * 100
                #    if len(m.agents) > 0
                #    else 0
                #),
            },
            agent_reporters={"agent_type": "type"}, #add more things to type, like economic state
        )

        # Create agents and place them on the grid
        for cell in self.grid.all_cells:
            if self.random.random() < self.density:
                #new types added with different values. The values were assigned to make the multiplier
                #function easier in the future
                agent_type = self.random.choices([1, 2, 3], weights = (scenario.frac1, scenario.frac2, scenario.frac3))[0]
                SchellingAgent(
                    self,
                    cell,
                    agent_type,
                    homophily=scenario.homophily,
                    radius=scenario.radius,
                )

        # Collect initial state
        self.agents.do("assign_state")
        self.datacollector.collect(self)

    def step(self):
        """Run one step of the model."""
        self.happy = 0  # Reset counter of happy agents
        self.agents.shuffle_do("step")  # Activate all agents in random order
        self.agents.do("assign_state")
        self.datacollector.collect(self)  # Collect data
        self.running = self.happy < len(self.agents)  # Continue until everyone is happy
