import math
from mesa.discrete_space import CellAgent
import numpy as np


class SchellingAgent(CellAgent):
    """
    Schelling Segregation Agent.
    The agent makes probabilistic neighborhood relocation choices based on housing affordability,
    local structural quality, and multi-scale demographic preferences, while dynamically 
    adapting their cooperation - defect strategy through localized social learning.
    """

    def __init__(self, model, cell, agent_type: int, income: int, agent_strategy: float, radius: int = 1,
        beta_mean: float = 1.0, beta_sigma: float = 1.0, utility_form: str = "continuous", 
        baseline_benefit: float = 1.0, move_cost: float = 0.5, logit_scale: float = 1.0, 
        budget_fraction: float = 0.5, quality_weight: float = 1.0, homophily_weight: float = 5.0, cost_weight: float = 1.0) -> None:
        """
        Create and initialize a new Schellingagent.

        Parameters:
        model : SchellingModel. The model instance to which this agent belongs.
        cell : Cell. The specific grid cell coordinate where this agent currently resides.
        agent_type : int. Income indicator upper class, medium class and lower class (e.g., 1, 2, or 3).
        income : int.The total income available to the agent.
        agent_strategy : float. Initial cooperation strategy, where 0.0 = full cooperator and 1.0 = full defector.
        radius : int, default 1. The micro-level search radius for checking immediate neighbor similarity.
        beta_mean : float, default 1.0. The baseline population median sensitivity to cost-benefit and budget penalties.
        beta_sigma : float, default 1.0. The standard deviation of cost sensitivity across the population.
        utility_form : str, default "continuous". Structural identifier toggle used to configure the active choice equations.
        baseline_benefit : float, default 1.0. The fundamental baseline utility derived from being housed in the system.
        move_cost : float, default 0.5. The behavioral friction or relocation penalty applied when changing locations.
        logit_scale : float, default 1.0. A normalization scalar translating raw financial calculations into utility.
        budget_fraction : float, default 0.5. The maximum fraction of income the agent is willing to spend on the rent.
        quality_weight : float, default 2.0. The structural preference weight assigned to neighborhood infrastructure and public goods.
        homophily_weight : float, default 2.0. The sociological preference weight assigned to neighborhood demographic similarity.

        Attributes:
        beta : float. The agent's individual price sensitivity coefficient, drawn from a log-normal distribution.
        strategy : float. The agent's strategy for cooperation. A value between 0.0 (full cooperator) and 1.0 (full defector).
        action : str. The agent's current action, either "C" (Cooperator) or "D" (Defector).
        contribution : float. The actual currency amount currently contributed to the local public goods pool.
        contribution_percentage : float. The fixed tax fraction of total income contributed if the agent is a cooperator (5%).
        current_utility : float. The agents current utility score, inspected by local peers for social learning.
        happy : bool. Flag indicating if the agent is happy.
        """
        super().__init__(model)
        self.cell = cell
        self.type = agent_type
        self.income = income
        self.strategy = agent_strategy
        self.radius = radius
        self.move_count = 0

        # Utility function parameters
        self.utility_form = utility_form
        self.baseline_benefit = baseline_benefit
        self.move_cost = move_cost
        self.logit_scale = logit_scale
        self.budget_fraction = budget_fraction
        self.quality_weight = quality_weight
        self.homophily_weight = homophily_weight
        self.cost_weight = cost_weight

        # Draw individual beta from a log-normal distribution to introduce heterogeneity in cost sensitivity
        # agent’s price sensitivity (how strongly they dislike high rent).
        # most agents are near beta_mean,
        # some are much more cost‑sensitive if beta_sigma is large.
        if beta_sigma > 0:
            z = self.model.random.gauss(0, 1)
            log_beta = math.log(beta_mean) + beta_sigma * z
            self.beta = math.exp(log_beta)
        else:
            self.beta = beta_mean

        # Choose initial cooperation strategy
        if self.strategy == 1.0:
            self.action = "D"
        else: 
            self.action = "C"

        self.contribution = 0.0
        self.contribution_percentage = 0.05 # Cooperators contribute 5% of their income to the neighborhood
        self.learning_rate = self.model.random.uniform(0, 1) # Heterogeneous learning rates for strategy updating
        self.current_utility = 0.0
        self.happy = False

    @property
    def neighbourhood(self):
        return self.model.cell_to_neighbourhood[self.cell.coordinate]
    
    @property
    def calculate_utility(self):
        self.current_utility = self.utility(self.neighbourhood, True)

    def contribute(self) -> None:
        """
        Calculate financial contribution based on strategy.
        If the agent is a cooperator, they contribute a fixed percentage of their income to the community.
        If the agent is a defector, their contribution is zero.
        """
        
        if self.model.random.random() < self.strategy:
            new_contribution = 0.0
            self.action = "D"  # Defector this round
        else:
            new_contribution = self.income * self.contribution_percentage
            self.action = "C"  # Cooperator this round
        self.neighbourhood.total_contribution += new_contribution - self.contribution
        self.contribution = new_contribution

    def choose_strategy(self):
        """
        Adapt to the strategy of the most successful neighbor if they are doing better than you.
        """
        neighbors = list(self.cell.get_neighborhood(radius=1).agents) # moore neighborhood
        if self.model.random.random() < 0.5 and len(neighbors) != 0:
            best_neighbor = max(neighbors, key=lambda a: a.current_utility)
        
            # Adapt your strategy towards the best neighbor's one weighted by your learning rate
            if best_neighbor.current_utility > self.current_utility:
                self.strategy = self.learning_rate*best_neighbor.strategy + (1-self.learning_rate)*self.strategy
        else:
            return

    def utility(self, neighbourhood, is_current: bool) -> float:
        '''
        Random Utility Theory: U_ij = V_ij + epsilon
        Utility from neighbourhood choice with homophily, cost burden, and moving costs.
        '''

        # macro level
        if neighbourhood.num_agents > 0: 
            similarity = neighbourhood.type_counts.get(self.type, 0) / neighbourhood.num_agents
        else:
            similarity = 0.0 

        # neighboorhood quality
        quality_benefit = self.quality_weight * neighbourhood.quality

        if self.action == 'C':
            actual_cost = neighbourhood.cost
        else:
            if neighbourhood.num_agents > 0:
                avg_income = sum(a.income for a in neighbourhood.agents) / neighbourhood.num_agents
            else:
                avg_income = self.income
            
            actual_cost = avg_income * self.model.base_rent

        rent = actual_cost / self.income

        if rent <= self.budget_fraction:
            affordability = 1.0
        elif rent >= 2 * self.budget_fraction:
            affordability = 0.0
        else:
            affordability = 1.0 - (rent - self.budget_fraction) / self.budget_fraction

        # moving 
        move_penalty = 0.0

        if not is_current:
            # counter 
            move_penalty = self.move_cost * (1.0 + 0.5 * self.move_count)

        # total utility = linear addition 
        V_ij = (self.baseline_benefit + 
                quality_benefit +
                affordability +
                (self.homophily_weight * similarity) - 
                (self.beta * self.cost_weight * rent) - 
                move_penalty
                )
        
        # scale = 1.0

        # U = 1.0 / (1.0 + math.exp(- scale * V_ij))

        return V_ij

    def step(self) -> None:
        """
        Relocation decision using multinomial logit over neighbourhoods.

        Choice set:
        - current neighbourhood is always available,
        - all other neighbourhoods that have at least one vacancy.

        Utilities:
        - computed by self.utility(neighbourhood, is_current),
        - is_current = True only for the current neighbourhood.
        """

        current_nb = self.neighbourhood

        # 1. Build choice set and utilities
        choice_set = []
        utilities = []

        # Current neighbourhood
        choice_set.append(current_nb)
        u_current = self.utility(current_nb, is_current=True)
        utilities.append(u_current)

        # Other neighbourhoods with at least one vacancy
        for nb in self.model.neighbourhoods.values():
            if nb is current_nb:
                continue

            if not nb.has_vacancy:
                continue

            choice_set.append(nb)
            u_nb = self.utility(nb, is_current=False)
            utilities.append(u_nb)

    # If no alternatives, only current nb is available, just stay and update realised utility
        if len(choice_set) == 1:
            self.current_utility = u_current
            return

        # Multinomial logit probabilities
        max_u = utilities[0]
        for u in utilities:
            if u > max_u:
                max_u = u

        weights = []
        for u in utilities:
            w = math.exp(self.logit_scale * (u - max_u))
            weights.append(w)

        chosen_nb = self.model.random.choices(
                                                choice_set, weights=weights, k=1)[0]

        # Move if a different neighbourhood was chosen
        new_possible_cell = []
        if chosen_nb is not current_nb:
            for cell in chosen_nb.cells:
                if cell.is_empty:
                    new_possible_cell.append(cell)

        if len(new_possible_cell) != 0:
            new_cell = self.model.random.choice(new_possible_cell)
            self.move_to(new_cell)

        # utility
        self.current_utility = self.utility(self.neighbourhood, is_current=True)

    def assign_state(self) -> None:
        """
        Update the model's global happiness counter based on the agent's final state.
        
        This method is called at the end of the simulation step after all agents 
        have made their choices.
        """
        # if self.happy:
        #     self.model.happy += 1

        if self.current_utility > 0.8:
            self.happy + 1

    def move_to(self, new_cell):
        old_nb = self.neighbourhood
        self.cell = new_cell
        new_nb = self.neighbourhood
        if old_nb != new_nb:
            old_nb.remove_agent(self)
            new_nb.add_agent(self)
            self.move_count +=1
