import math
from mesa.discrete_space import CellAgent


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
        budget_fraction: float = 0.5, quality_weight: float = 2.0, homophily_weight: float = 2.0, cost_weight: float = 3.0) -> None:
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
        if beta_sigma > 0:
            z = self.model.random.gauss(0, 1)
            log_beta = math.log(beta_mean) + beta_sigma * z
            self.beta = math.exp(log_beta)
        else:
            self.beta = beta_mean

        # Choose initial cooperation strategy
        if self.model.random.random() < self.model.defector_frac:
            self.strategy = "D"
        else:
            self.strategy = "C"

        self.contribution = 0.0
        self.contribution_percentage = 0.05
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
        if self.strategy == "D":
            new_contribution = 0.0
            #self.contribution = 0.0
        else:
            new_contribution = self.income * self.contribution_percentage
            #self.contribution = self.income * self.contribution_percentage
        self.neighbourhood.total_contribution += new_contribution - self.contribution
        self.contribution = new_contribution

    def choose_strategy(self):
        """
        Copy the strategy of the most successful neighbor by a 50% chance.
        """
        neighbors = list(self.cell.get_neighborhood(radius=1).agents) # moore neighborhood

        if not neighbors:
            return

        best_neighbor = max(neighbors, key=lambda a: a.current_utility)
       
        if best_neighbor.current_utility > self.current_utility:
            self.strategy = best_neighbor.strategy

    def utility(self, neighbourhood, is_current: bool) -> float:
        """
        Cost-benefit utility of one neighbourhood for this agent.
        The utility is a function of the neighbourhood's cost, quality, and demographic composition,
        as well as the agent's income, preferences, and the potential move cost if it's not the current neighbourhood.
        """
        price = neighbourhood.cost
        burden = price / self.income
        excess = burden - self.budget_fraction

        if excess > 0:
            penalty = excess
        else:
            penalty = 0.0

        num_local_agents = neighbourhood.num_agents

        if num_local_agents > 0:
            macro_score = neighbourhood.type_counts.get(self.type) / num_local_agents
        else:
            macro_score = 1.0

        
        if is_current:
            micro_cell = self.cell.get_neighborhood(radius=self.radius)
            micro_agents = []

            for cell in micro_cell:
                if not cell.is_empty:
                    for neighbor_agent in cell.agents:
                        micro_agents.append(neighbor_agent)

            if micro_agents:
                micro_same_type_count = 0
                for agent in micro_agents:
                    if agent.type == self.type:
                        micro_same_type_count += 1
                
                homophily_score = micro_same_type_count / len(micro_agents)
            else:   
                homophily_score = 1.0
        else:
            homophily_score = macro_score

        # neighborhood's attractiveness
        dynamic_quality = neighbourhood.quality

        total_utility = (self.baseline_benefit + 
                         (self.quality_weight * dynamic_quality) +
                         (self.homophily_weight * homophily_score) +  
                         (0.5 * self.homophily_weight) * macro_score
                        )
        
        total_utility -= self.beta * self.logit_scale * penalty

        if not is_current:
            total_utility -= self.move_cost
        total_utility -= self.cost_weight * (self.contribution / self.income)
        return total_utility 

    def step(self) -> None:
        """
        Make a conditional choice over the available neighbourhoods.
        """
        current_neighbourhood = self.neighbourhood
        # Choose between the current neighbourhood and every neighbourhood with a vacancy.
        utilities = []
        choice_set = [current_neighbourhood]
        for neighbourhood in self.model.neighbourhoods.values():
            if neighbourhood.id == current_neighbourhood.id:
                is_current = True
                utilities.append(self.utility(neighbourhood, is_current))
                continue
            for cell in neighbourhood.cells:
                if cell.is_empty:
                    choice_set.append(neighbourhood)
                    # Score each neighbourhood in the choice set.
                    is_current = False
                    utilities.append(self.utility(neighbourhood, is_current))
                    break

    
        #utilities = []
        #for neighbourhood in choice_set:
            #if neighbourhood.id == current_neighbourhood.id:
                #is_current = True
            #else:
                #is_current = False
            #utilities.append(self.utility(neighbourhood, is_current))

        max_utility = max(utilities)
        choice_weights = []
        for utility in utilities:
            choice_weights.append(math.exp(utility - max_utility))

        # Randomly pick one neighbourhood, weighted by the scores.
        chosen_neighbourhood = self.model.random.choices(choice_set, weights=choice_weights)[0]

        # If the agent chose to stay in its current neighbourhood, determine if it's happy or not.
        if chosen_neighbourhood.id == current_neighbourhood.id:
            price = current_neighbourhood.cost
            burden = price / self.income
            
            #print(f"Cost={price}, Income={self.income}, Threshold={self.income * self.budget_fraction}")
            
            if burden <= self.budget_fraction:
                self.happy = True
            else:
                self.happy = False
            self.current_utility = self.utility(current_neighbourhood, is_current=True)

        else:
            # Otherwise move into one of the empty cells of the chosen neighbourhood.
            empty_cells = []

            for cell in chosen_neighbourhood.cells:
                if cell.is_empty:
                    empty_cells.append(cell)

            if empty_cells:
                new_cell = self.model.random.choice(empty_cells)
                self.move_to(new_cell)
                self.happy = False #the agent isnt settled and is not yet happy
            else:
                # The chosen neighbourhood is full, the agent is unhappy.
                self.happy = False
            
            self.current_utility = self.utility(current_neighbourhood, is_current=False)
            

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
    # def assign_state(self) -> None:
    #     """Determine if agent is happy and move if necessary."""
    #     neighbors = list(self.cell.get_neighborhood(radius=self.radius).agents)

    #     # Count similar neighbors
    #     similar_neighbors = len([n for n in neighbors if n.type == self.type])
    #     costs = {nb.id: nb.cost for nb in self.model.neighbourhoods.values()}

    #     # Calculate the fraction of similar neighbors
    #     if (valid_neighbors := len(neighbors)) > 0:
    #         similarity_fraction = similar_neighbors / valid_neighbors
    #     else:
    #         # If there are no neighbors, the similarity fraction is 0
    #         similarity_fraction = 0.0
    #     if self.type == 3:
    #         highest_cost = max(n.cost for n in self.model.neighbourhoods.values())
    #         higher_exists = highest_cost > self.neighbourhood.cost
    #         if higher_exists and self.model.random.random() < 0.3:
    #             self.happy = False
    #         else:
    #             self.happy = True
    #             self.model.happy += 1
    #         #self.happy = self.neighbourhood.cost >= highest_cost
    #     else:
    #         if self.neighbourhood.cost <= self.income:
    #             self.happy = True
    #             self.model.happy += 1
    #         else:
    #             self.happy = False
    #     #if similarity_fraction < self.homophily:
    #         #self.happy = False
    #     #else:
    #         #self.happy = True
    #         #self.model.happy += 1

    # def step(self) -> None:
    #     # Move if unhappy
    #     if self.happy: 
    #         return
    #     if self.type == 3:
    #         higher_nb = [nb for nb in self.model.neighbourhoods.values() if nb.cost > self.neighbourhood.cost]
    #         candidate_nbs = [c for n in higher_nb for c in n.cells if c.is_empty]
    #         if candidate_nbs:
    #             self.cell = self.model.random.choice(candidate_nbs)
    #     else:
    #         lower_nb = [nb for nb in self.model.neighbourhoods.values() if nb.cost < self.neighbourhood.cost]
    #         candidate_nbs = [c for n in lower_nb for c in n.cells if c.is_empty]
    #         if candidate_nbs:
    #             self.cell = self.model.random.choice(candidate_nbs)
    #         #affordable = [nb for nb in self.model.neighbourhoods.values() if nb.cost <= self.income]
    #         #empty = [c for n in affordable for c in n.cells if c.is_empty]
    #         #if empty:
    #         #    self.cell = self.model.random.choice(empty)
