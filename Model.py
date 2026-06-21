import math
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from Agents import SchellingAgent
from mesa.experimental.scenarios import Scenario
from mesa.discrete_space import PropertyLayer
from Convergence_discrete import compute_H
import numpy as np
from Clustering import compute_morans_I, compute_neighbourhood_income_variance

class Neighbourhood:
    def __init__(self, id, model, seed_coord):
        self.id = id
        self.model = model
        self.cells = []
        self.seed = seed_coord
        self.cost = 0.0
        self.num_agents = 0
        self.type_counts = {1: 0, 2: 0, 3: 0}
        self.total_contribution = 0.0

    @property
    def agents(self):
        agents_list = []
        for cell in self.cells:
            for agent in cell.agents:
                agents_list.append(agent)
        return agents_list
    
    def remove_agent(self, agent):
        self.num_agents -= 1
        self.type_counts[agent.type] -= 1
        self.total_contribution -= agent.contribution

    def add_agent(self, agent):
        self.num_agents += 1
        self.type_counts[agent.type] += 1
        self.total_contribution += agent.contribution

    @property
    def avg_contribution(self):
        return self.total_contribution / self.num_agents if self.num_agents != 0 else 0.0
    
    @property 
    def quality(self):
        if self.num_agents == 0:
            return 0.0
        return self.avg_contribution / self.model.max_contribution #normalize to [0,1]
    
    @property 
    def has_vacancy(self):
        return self.num_agents < len(self.cells)

    #def update_quality(self):
        #"""
        #Contribution of agents to the neighboorhood quality.
        #"""
       #current_residents = self.agents
        #total_contributions = 0.0

        #for agent in current_residents:
            #total_contributions += agent.contribution
                
        # Neighborhood quality becomes the average contribution per resident
        #residents_count = len(current_residents)

        #if residents_count > 0:
            #self.quality = total_contributions / residents_count
        
        #else:
            #self.quality = 0.0

    def update_cost(self, adjust = 0.3):

        current_residents = self.agents

        if len(current_residents) > 0:
            total_income = 0

            for agent in current_residents:
                total_income += agent.income

            avg_income = total_income / len(current_residents)
            rent_fraction = self.model.base_rent + self.model.quality_premium * self.quality
            target = avg_income * rent_fraction
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
    seed: int = 69
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
    learning_rate: float = 0.5
    
    # Mixed-logit parameters
    beta_mean: float = 1.0  # population mean      
    beta_sigma: float = 1.0 # heterogeneity
    utility_form: str = "continuous" # threshold 
    
    # Cost-benefit utility parameters
    baseline_benefit: float = 1.0 # utility of being housed
    move_cost: float = 0.5 # penalty for relocating
    logit_scale: float = 1.0 # converts the burden penalty into utility units
    budget_fraction: float = 0.5 # fraction of income an agent will spend on housing
    base_rent: float = 0.3 # baseline rent as a fraction of local income
    quality_premium: float = 0.4 # extra rent fraction a top-quality neighbourhood charges
    quality_weight: float = 2.0 # how much agents value a neighbourhood's quality
    cost_weight: float = 3.0 # how painful is the cost of cooperating to an agent
    activation_rate: float = 0.3 # how often the agent is activated per poisson process rules
    step_count: int = 0

class Schelling(Model):
    """Model class for the Schelling segregation model."""

    def __init__(self, scenario: SchellingScenario = SchellingScenario):
        """Create a new Schelling model.

        Args:
            scenario: SchellingScenario containing model parameters.
        """
        super().__init__(scenario=scenario)
        self.random.seed(int(scenario.seed))

        # Model parameters
        self.density = scenario.density
        self.frac1 = scenario.frac1
        self.frac2 = scenario.frac2
        self.frac3 = max(0.0, 1.0 - self.frac1 - self.frac2)
        self.base_rent = scenario.base_rent
        self.quality_premium = scenario.quality_premium
        self.defector_frac = scenario.defector_frac
        self.activation_rate = scenario.activation_rate
        self.step_count = scenario.step_count

        # added a learning rate here
        self.learning_rate = scenario.learning_rate

        # added homophily score
        self.homophily = scenario.homophily

        # Segregation tracking
        self.H_history = [] # tracking H values
        self.epsilon = 1e-2 # convergence threshold
        self.convergence_window = 50 # number of steps that H must be stable for to call it 'convergence'

        # util history
        self.utility_history = []  # list of {"median": ..., "q25": ..., "q75": ...}

        # Segregation tracking
        self.H_history = [] # tracking H values
        self.epsilon = 1e-2 # convergence threshold
        self.convergence_window = 50 # number of steps that H must be stable for to call it 'convergence'

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
                "defector_proportion": lambda m: (sum(agent.action == "D" for agent in m.agents) / len(m.agents) if len(m.agents) > 0 else 0),
                "mean_defection_probability": lambda m: (sum(agent.strategy for agent in m.agents) / len(m.agents) if len(m.agents) > 0 else 0),
                "H": lambda m: m.H_history[-1] if m.H_history else None, # convergence metric
                "morans_I": lambda m: compute_morans_I(m), # clustering metric
                "nb_income_variance": lambda m: compute_neighbourhood_income_variance(m), # income variance
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

                # agents initialised with a strategy based on the defector fraction set
                agent_strategy = self.random.choices([0.0, 1.0], weights = (1 - scenario.defector_frac, scenario.defector_frac))[0]
                SchellingAgent(
                    self,
                    cell,
                    agent_type,
                    income,
                    agent_strategy,
                    beta_mean = scenario.beta_mean,
                    beta_sigma = scenario.beta_sigma,
                    utility_form = scenario.utility_form,
                    radius = scenario.radius,
                    baseline_benefit = scenario.baseline_benefit,
                    move_cost = scenario.move_cost,
                    logit_scale = scenario.logit_scale,
                    budget_fraction = scenario.budget_fraction,
                    quality_weight = scenario.quality_weight,
                    cost_weight= scenario.cost_weight,
                )
        self.max_contribution = max(agent.income for agent in self.agents) * 0.05
        self.add_neighbourhood_values()
        for i in self.neighbourhoods.values():
            i.update_cost()

        # Collect initial state
        self.agents.do("assign_state")
        self.datacollector.collect(self)

    def add_neighbourhood_values(self):
        for nb in self.neighbourhoods.values():
            nb.num_agents = 0
            nb.type_counts = {1: 0, 2: 0, 3: 0}
            nb.total_contribution = 0.0
        for agent in self.agents:
            agent.neighbourhood.add_agent(agent)

    def build_neighbourhoods(self, neighbourhood_count):
        all_cells = list(self.grid.all_cells)
        seeds = self.random.sample(all_cells, neighbourhood_count)

        seed_coords = []
        for seed in seeds:
            seed_coords.append(seed.coordinate)

        self.neighbourhoods = {}
        for index in range(neighbourhood_count):
            quality = self.random.random()
            self.neighbourhoods[index] = Neighbourhood(index, self, seed_coords[index])
        self.cell_to_neighbourhood = {}

        for cell in all_cells:
            x, y = cell.coordinate
            nearest_index = None
            best_distance = math.inf
            for index in range(neighbourhood_count):
                distance = (x - seed_coords[index][0])**2 + (y - seed_coords[index][1])**2
                if distance < best_distance:
                    best_distance = distance
                    nearest_index = index
            self.nb_layer.data[x, y] = nearest_index
            neighbourhood = self.neighbourhoods[nearest_index]
            neighbourhood.cells.append(cell)
            self.cell_to_neighbourhood[cell.coordinate] = neighbourhood


    def step(self):
        """
        Run one step of the model.
        """
        self.happy = 0  # Reset counter of happy agents
        self.agents.do("contribute")
        for i in self.neighbourhoods.values():
            i.update_cost()
        #discrete poisson process with bernoulli per step
        agents = list(self.agents)
        self.random.shuffle(agents)
        for a in agents:
            if self.random.random() < self.activation_rate:
                a.step()
            else:
                a.calculate_utility 
        self.agents.shuffle_do("choose_strategy")
        self.agents.do("assign_state")

        # Segregation metric H
        H = compute_H(
            list(self.neighbourhoods.values()),
            list(self.agents)
        )
        self.H_history.append(H)

        utilities = [a.current_utility for a in self.agents]
        self.utility_history.append({
            "median": float(np.median(utilities)),
            "q25":    float(np.percentile(utilities, 25)),
            "q75":    float(np.percentile(utilities, 75)),
        })

        self.datacollector.collect(self) # Collect data
        self.step_count += 1
        # Convergence check
        # Stop if everyone is happy OR if H has been stable for some number 'convergence_window' of steps
        segregation_converged = (
            len(self.H_history) >= self.convergence_window
            and (max(self.H_history[-self.convergence_window:])
                - min(self.H_history[-self.convergence_window:])) < self.epsilon
        ) and self.step_count >= 150
        self.running = not segregation_converged and self.step_count < 1000  # Continue until everyone is happy or H stable
