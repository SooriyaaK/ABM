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

    def __init__(self, model, cell, agent_type: int, income: int, radius: int = 1,
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
        strategy : str. Active cooperative strategy, either C (Cooperator) or D (Defector).
        contribution : float. The actual currency amount currently contributed to the local public goods pool.
        contribution_percentage : float. The fixed tax fraction of total income contributed if the agent is a cooperator (5%).
        current_utility : float. The agents current utility score, inspected by local peers for social learning.
        happy : bool. Flag indicating if the agent is happy.
        """
        super().__init__(model)
        self.cell = cell
        self.type = agent_type
        self.income = income
        self.radius = radius

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
        # if beta_sigma > 0:
        #     z = self.model.random.gauss(0, 1)
        #     log_beta = math.log(beta_mean) + beta_sigma * z
        #     self.beta = math.exp(log_beta)
        # else:
        #     self.beta = beta_mean

        # Choose initial cooperation strategy
        if self.model.random.random() < self.model.defector_frac:
            self.strategy = 1.0 # Chance of defecting 100%
        else:
            self.strategy = 0.0 # Chance of defecting 0%

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

        # moving 
        if not is_current:
            # counter 
            move_penalty = self.move_cost
        else:
            move_penalty = 0.0

        # total utility = linear addition 
        V_ij = (self.baseline_benefit + 
                quality_benefit +
                (self.homophily_weight * similarity) - 
                (self.cost_weight * rent) - 
                move_penalty
                )

        return V_ij
    
    # def step(self) -> None:
    #     """
    #     Agent's decision step: choose neighbourhood, evaluate happiness, update utility.
        
    #     1. Build choice set (current + neighbourhoods with vacancies)
    #     2. Calculate utilities for each option using mixed logit
    #     3. Make probabilistic choice (multinomial logit)
    #     4. Determine happiness (affordability + homophily)
    #     5. Move if chose different neighbourhood
    #     6. Update current utility based on actual location
    #     """
    
    #     current_neighbourhood = self.neighbourhood
    
    #     # set of empty houses in the current neighbourhood and other neighbourhood
    #     utilities = []
    #     choice_set = [current_neighbourhood]
    
    #     for neighbourhood in self.model.neighbourhoods.values():
    #         if neighbourhood.id == current_neighbourhood.id:
    #             is_current = True
    #             utilities.append(self.utility(neighbourhood, is_current))
    #             continue
    #         for cell in neighbourhood.cells:
    #             if cell.is_empty:
    #                 choice_set.append(neighbourhood)
    #                 is_current = False
    #                 utilities.append(self.utility(neighbourhood, is_current))
    #                 break

    #     # multimodal logit choice
    #     max_utility = max(utilities)
    #     choice_weights = []
    #     for utility in utilities:
    #         choice_weights.append(math.exp(utility - max_utility))

    #     # Randomly pick one neighbourhood, weighted by the scores.
    #     chosen_neighbourhood = self.model.random.choices(choice_set, weights=choice_weights)[0]

    #     # 
    #     if chosen_neighbourhood.id == current_neighbourhood.id:
    #         #if the agent decides to stay in the current neighbourhood
    #         price = current_neighbourhood.cost
    #         burden = price / self.income
    #         affordable = burden <= self.budget_fraction

    #         # Homophily micro-level
    #         micro_cell = self.cell.get_neighborhood(radius=self.radius)
    #         micro_agents = []

    #         # Collect neighbors
    #         for cell in micro_cell:
    #             if not cell.is_empty:
    #                 for agent in cell.agents:
    #                     micro_agents.append(agent)
            
    #         # Count similar types
    #         if micro_agents:
    #             micro_same_type_count = 0
    #             for agent in micro_agents:
    #                 if agent.type == self.type:
    #                     micro_same_type_count += 1
                
    #             homophily_score = micro_same_type_count / len(micro_agents)
    #             homophilic = homophily_score >= 0.4  # Adjust threshold
    #         else:
    #             homophilic = True
            
    #         # Happy if both affordability and same type of agent
    #         self.happy = affordable and homophilic
            
    #         # Update utility for current location 
    #         self.current_utility = self.utility(current_neighbourhood, is_current=True)

    #     else:
    #         # if the agent choose to move to a different neighboorhood
    #         # Otherwise move into one of the empty cells of the chosen neighbourhood.
    #         empty_cells = []

    #         for cell in chosen_neighbourhood.cells:
    #             if cell.is_empty:
    #                 empty_cells.append(cell)

    #         if empty_cells:
    #             # Move agent to new neighbourhood
    #             new_cell = self.model.random.choice(empty_cells)
    #             self.move_to(new_cell)
    #             self.happy = False  # Agent isn't settled and is not yet happy
                
    #             # Update utility for new neighbourhood where agent now is
    #             self.current_utility = self.utility(self.neighbourhood, is_current=True)
    #         else:
    #             # The chosen neighbourhood is full, the agent is unhappy and stays
    #             self.happy = False
                
    #             # Agent remains in current neighbourhood
    #             self.current_utility = self.utility(current_neighbourhood, is_current=True)



    def assign_state(self) -> None:
        """
        Update the model's global happiness counter based on the agent's final state.
        
        This method is called at the end of the simulation step after all agents 
        have made their choices.
        """
        if self.happy:
            self.model.happy += 1

    def move_to(self, new_cell):
        old_nb = self.neighbourhood
        self.cell = new_cell
        new_nb = self.neighbourhood
        if old_nb != new_nb:
            old_nb.remove_agent(self)
            new_nb.add_agent(self)
