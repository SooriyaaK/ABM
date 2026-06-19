import math
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from Agents import SchellingAgent
from mesa.experimental.scenarios import Scenario
from mesa.discrete_space import PropertyLayer


class Neighbourhood:
    def __init__(self, id, cells, seed_coord):
        self.id = id
        self.cells = cells
        self.cost = 0.0
        self.seed = seed_coord

    @property
    def agents(self):
        agents_list = []
        for cell in self.cells:
            for a in cell.agents:
                agents_list.append(a)
        return agents_list
    
    def update_cost(self, adjust = 0.3):

        current_residents = self.agents

        if len(current_residents) > 0:
            total_income = 0

            for agent in current_residents:
                total_income += agent.income

            avg_income = total_income / len(current_residents)
            occupancy = self.num_agents / len(self.cells)
            rent_fraction = self.model.base_rent + self.model.quality_premium * self.quality
            target = avg_income * rent_fraction * (1 + occupancy)
            self.cost += adjust * (target - self.cost) #prices adjust slowly, not instantly

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
    neighbourhood_count: int = 25
    defector_frac: float = 0.5
    
    # Mixed-logit parameters
    beta_mean: float = 1.0  # population mean      
    beta_sigma: float = 1.0 # heterogeneity
    utility_form: str = "continuous" # threshold 
    
    # Cost-benefit utility parameters
    baseline_benefit: float = 1.0 # utility of being housed
    move_cost: float = 0.5 # penalty for relocating
    logit_scale: float = 1.0 # converts the burden penalty into utility units
    budget_fraction: float = 0.3 # fraction of income an agent will spend on housing
    base_rent: float = 0.05 # baseline rent as a fraction of local income
    quality_premium: float = 0.2 # extra rent fraction a top-quality neighbourhood charges
    quality_weight: float = 2.0 # how much agents value a neighbourhood's quality
    cost_weight: float = 3.0 # how painful is the cost of cooperating to an agent
    activation_rate: float = 0.3 # how often the agent is activated per poisson process rules

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
            (scenario.width, scenario.height), random=self.random, capacity=1
        )
        self.nb_layer = PropertyLayer("neighbourhood", (scenario.width, scenario.height), default_value=0, dtype=int)
        self.grid.add_property_layer(self.nb_layer)

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

        self.build_neighbourhoods(scenario.neighbourhood_count)

        # Create agents and place them on the grid
        for cell in self.grid.all_cells:
            if self.random.random() < self.density:
                #new types added with different values. The values were assigned to make the multiplier
                #function easier in the future
                agent_type = self.random.choices([1, 2, 3], weights = (scenario.frac1, scenario.frac2, scenario.frac3))[0]
                if agent_type == 1:
                    income = 1000
                elif agent_type == 2:
                    income = 2000
                else:
                    income = 4000
                SchellingAgent(
                    self,
                    cell,
                    agent_type,
                    income,
                    homophily=scenario.homophily,
                    radius=scenario.radius,
                )
        for i in self.neighbourhoods.values():
            i.update_cost()
        # Collect initial state
        self.agents.do("assign_state")
        self.datacollector.collect(self)

    def build_neighbourhoods(self, n):
        all_cells = list(self.grid.all_cells)
        seeds = self.random.sample(all_cells, n)
        seed_coords = [s.coordinate for s in seeds]

        self.neighbourhoods = {i: Neighbourhood(i, [], seed_coords[i]) for i in range(n)}
        self.cell_to_neighbourhood = {}

        for cell in all_cells:
            x, y = cell.coordinate
            best_seed = None
            best_dist = math.inf
            for i in range(n):
                dist = (x - seed_coords[i][0])**2 + (y - seed_coords[i][1])**2
                if dist < best_dist:
                    best_dist = dist
                    best_seed = i
            nid = best_seed
            self.nb_layer.data[x, y] = nid
            nb = self.neighbourhoods[nid]
            nb.cells.append(cell)
            self.cell_to_neighbourhood[cell.coordinate] = nb


    def step(self):
        """Run one step of the model."""
        self.happy = 0  # Reset counter of happy agents
        self.agents.shuffle_do("step")  # Activate all agents in random order
        self.agents.do("assign_state")
        for i in self.neighbourhoods.values():
            i.update_cost()
        self.datacollector.collect(self)  # Collect data
        self.running = self.happy < len(self.agents)  # Continue until everyone is happy
