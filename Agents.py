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
        if self.model.random.random() < 0.5 and len(neighbors) != 0:
            best_neighbor = max(neighbors, key=lambda a: a.current_utility)
        
            if best_neighbor.current_utility > self.current_utility:
                self.strategy = best_neighbor.strategy
        else:
            return

    # def utility(self, neighbourhood, is_current: bool) -> float:
    #     """
    #     Cost-benefit utility of one neighbourhood for this agent.
    #     The utility is a function of the neighbourhood's cost, quality, and demographic composition,
    #     as well as the agent's income, preferences, and the potential move cost if it's not the current neighbourhood.
    #     """
    #     price = neighbourhood.cost
    #     burden = price / self.income
    #     excess = burden - self.budget_fraction

    #     if excess > 0:
    #         penalty = excess
    #     else:
    #         penalty = 0.0

    #     num_local_agents = neighbourhood.num_agents

    #     if num_local_agents > 0:
    #         macro_score = neighbourhood.type_counts.get(self.type) / num_local_agents
    #     else:
    #         macro_score = 1.0

        
    #     if is_current:
    #         micro_cell = self.cell.get_neighborhood(radius=self.radius)
    #         micro_agents = []

    #         for cell in micro_cell:
    #             if not cell.is_empty:
    #                 for neighbor_agent in cell.agents:
    #                     micro_agents.append(neighbor_agent)

    #         if micro_agents:
    #             micro_same_type_count = 0
    #             for agent in micro_agents:
    #                 if agent.type == self.type:
    #                     micro_same_type_count += 1
                
    #             homophily_score = micro_same_type_count / len(micro_agents)
    #         else:   
    #             homophily_score = 1.0
    #     else:
    #         homophily_score = macro_score

    #     # neighborhood's attractiveness
    #     dynamic_quality = neighbourhood.quality

    #     total_utility = (self.baseline_benefit + 
    #                      (self.quality_weight * dynamic_quality) +
    #                      (self.homophily_weight * homophily_score) +  
    #                      (0.5 * self.homophily_weight) * macro_score
    #                     )
        
    #     total_utility -= self.beta * self.logit_scale * penalty

    #     if not is_current:
    #         total_utility -= self.move_cost
    #     total_utility -= self.cost_weight * (self.contribution / self.income)
    #     return total_utility 

    def utility(self, neighbourhood, is_current: bool) -> float:
        """
        Random utility model with heterogeneous cost sensitivity beta,
        cost-benefit utility of a neighbourhood for this agent.
        
        This function evaluates how attractive a neighbourhood is based on:
        - Affordability: housing cost relative to income.
        - Quality: average contribution level of residents.
        - Social factors: demographic similarity to neighbors.
        - Cooperation costs: agent's own contribution burden.
        - Relocation: cost of moving.
        
        The utility combines these into a single number. Higher utility = agent prefers this neighbourhood.
        
        Args:
            neighbourhood: Neighbourhood object being evaluated.
            is_current: bool. True if this is agent's current neighbourhood.
        
        Returns: float: Systematic utility V_ij (before random error term).
        """


        # How much of the agent's income does housing cost?
        price = neighbourhood.cost
        burden = price / self.income  # Normalized price (fraction of income)
    
        # Linear cost penalty: heterogeneous sensitivity beta_i varies by agent
        # Agents with high beta are very price-sensitive, a low beta means less sensitive
        cost_penalty = self.beta * self.logit_scale * burden
    
        # Affordability penalty for exceeding budget
        excess = burden - self.budget_fraction
        if excess > 0:
            # Above budget: increases quadratically, making it very painful
            affordability_cliff = 2.0 * (excess ** 2)
        else:
            # Under budget: no additional penalty
            affordability_cliff = 0.0
    
        # calc the total pain of the housing cost
        total_cost_disutility = cost_penalty + affordability_cliff
    
        # How good is the neighbourhood? 
        # Measured by average contribution of residents.
        # neighbourhood.quality ranges from [0, 1] - a normalized contribution
        # Higher quality means agents value living here more

        dynamic_quality = neighbourhood.quality
        
        # Quality utility: all agents value quality equally: no heterogeneity here
        quality_utility = self.quality_weight * dynamic_quality
    
        # Agents want to live near similar types (homophily = preference for similar)
    
        # similarity on a neighbourhood-level: macro demographic
        num_local_agents = neighbourhood.num_agents
    
        if num_local_agents > 0:
            # What fraction of the neighbourhood is the same type as me?
            macro_score = neighbourhood.type_counts.get(self.type) / num_local_agents
        else:
            # Empty neighbourhood: treat as neutral
            macro_score = 1.0
    

        if is_current:
            # current neighboorhood: check on a micro level the 8 immediate neighbors in radius
            micro_cell = self.cell.get_neighborhood(radius=self.radius)
            micro_agents = []
            
            # Collect all agents in the micro search radius
            for cell in micro_cell:
                if not cell.is_empty:
                    for neighbor_agent in cell.agents:
                        micro_agents.append(neighbor_agent)
        
            # Calculate micro-level similarity
            if micro_agents:
                # Count how many neighbors are the same type as me
                micro_same_type_count = sum(1 for agent in micro_agents if agent.type == self.type)
                homophily_score = micro_same_type_count / len(micro_agents)
            else:   
                # No neighbors: treat as neutral
                homophily_score = 1.0
        else:
            # different neighbourhood: macro level - agents don't know their immediate neighbour before moving in
            homophily_score = macro_score
    
    
        # Heterogeneous sensitivity to homophily
        # Agents with high beta care more about demographic similarity
        # Some agents are more socially-driven, others less so
        homophily_utility = self.beta * self.homophily_weight * homophily_score
        
        # calc the broader demographic composition 
        macro_utility = 0.5 * self.homophily_weight * macro_score
        
        # Cooperators contribute 5% of income to neighbourhood quality
        # Defectors don't contribute, so they don't pay this cost
        
        cooperation_cost = 0.0
        if self.strategy == "C":
            cooperation_cost = self.cost_weight * (self.contribution / self.income)
     

        # Moving to a new neighbourhood costs
        # Agents prefer to stay 

        move_friction = 0.0
        if not is_current:
            # Only penalize non-current neighbourhoods (candidate moves)
            # Current neighbourhood has zero friction (already here)
            move_friction = self.move_cost
        
        
        # total utility
        # Higher utility = agent prefers this neighbourhood
        
        total_utility = (
            # Base utility of being housed 
            self.baseline_benefit
            
            # Subtract all costs
            - total_cost_disutility       # pain? Costs money? Utility goes down
            - cooperation_cost            # Cooperator? Pays tax, utility goes down
            - move_friction               # Moving to new place? utility goes down
            
            # Add all benefits
            + quality_utility             # Nice neighbourhood? Utility goes up
            + homophily_utility           # Similar neighbors? Utility goes up
            + macro_utility               # Broader demographic fit? Utility goes up
        )
        
        return total_utility
    
    def step(self) -> None:
        """
        Agent's decision step: choose neighbourhood, evaluate happiness, update utility.
        
        1. Build choice set (current + neighbourhoods with vacancies)
        2. Calculate utilities for each option using mixed logit
        3. Make probabilistic choice (multinomial logit)
        4. Determine happiness (affordability + homophily)
        5. Move if chose different neighbourhood
        6. Update current utility based on actual location
        """
    
        current_neighbourhood = self.neighbourhood
    
        # set of empty houses in the current neighbourhood and other neighbourhood
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
                    is_current = False
                    utilities.append(self.utility(neighbourhood, is_current))
                    break

        # multimodal logit choice
        max_utility = max(utilities)
        choice_weights = []
        for utility in utilities:
            choice_weights.append(math.exp(utility - max_utility))

        # Randomly pick one neighbourhood, weighted by the scores.
        chosen_neighbourhood = self.model.random.choices(choice_set, weights=choice_weights)[0]

        # 
        if chosen_neighbourhood.id == current_neighbourhood.id:
            #if the agent decides to stay in the current neighbourhood
            price = current_neighbourhood.cost
            burden = price / self.income
            affordable = burden <= self.budget_fraction

            # Homophily micro-level
            micro_cell = self.cell.get_neighborhood(radius=self.radius)
            micro_agents = []

            # Collect neighbors
            for cell in micro_cell:
                if not cell.is_empty:
                    for agent in cell.agents:
                        micro_agents.append(agent)
            
            # Count similar types
            if micro_agents:
                micro_same_type_count = 0
                for agent in micro_agents:
                    if agent.type == self.type:
                        micro_same_type_count += 1
                
                homophily_score = micro_same_type_count / len(micro_agents)
                homophilic = homophily_score >= 0.4  # Adjust threshold
            else:
                homophilic = True
            
            # Happy if both affordability and same type of agent
            self.happy = affordable and homophilic
            
            # Update utility for current location 
            self.current_utility = self.utility(current_neighbourhood, is_current=True)

        else:
            # if the agent choose to move to a different neighboorhood
            # Otherwise move into one of the empty cells of the chosen neighbourhood.
            empty_cells = []

            for cell in chosen_neighbourhood.cells:
                if cell.is_empty:
                    empty_cells.append(cell)

            if empty_cells:
                # Move agent to new neighbourhood
                new_cell = self.model.random.choice(empty_cells)
                self.move_to(new_cell)
                self.happy = False #the agent isnt settled and is not yet happy
                self.current_utility = self.utility(self.neighbourhood, is_current=False)
                # Update utility for new neighbourhood where agent now is
            else:
                # The chosen neighbourhood is full, the agent is unhappy and stays
                self.happy = False
                self.current_utility = self.utility(self.neighbourhood, is_current=True)
            

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
