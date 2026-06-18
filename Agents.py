import math
from mesa.discrete_space import CellAgent


class SchellingAgent(CellAgent):
    """
    Schelling Segregation Agent.
    Each agent has randomlt assigned an income, based on that income a housing cost is calculated.
    The agent then makes a choice of neighbourhood to live in, based on the cost and their income.
    """

    def __init__(self, model, cell, agent_type: int, income: int, beta_mean: float = 1.0, beta_sigma: float = 1.0, utility_form: str = "continuous", radius: int = 1, baseline_benefit: float = 1.0, move_cost: float = 0.5, logit_scale: float = 1.0, budget_fraction: float = 0.5, quality_weight: float = 2.0) -> None:
        """
        Create a new Schelling agent.

        parameters:
        model: The model instance the agent belongs to.
        cell: The grid cell the agent starts on.
        agent_type: Income indicator (1, 2, or 3).
        income: Agent's income.
        beta_mean: Sensitivity to cost-benefit utility. Population median.
        beta_sigma: Spread of the sensitivity to cost-benefit utility.
        utility_form: .
        radius: Search radius.
        baseline_benefit: Utility of being housed.
        move_cost: Penalty for relocating.
        logit_scale: Converts money units into utility units.
        """
        super().__init__(model)
        self.cell = cell
        self.type = agent_type
        self.income = income
        self.radius = radius

        self.utility_form = utility_form
        self.baseline_benefit = baseline_benefit
        self.move_cost = move_cost
        self.logit_scale = logit_scale
        self.budget_fraction = budget_fraction
        self.quality_weight = quality_weight

        if beta_sigma > 0:
            z = self.model.random.gauss(0, 1)
            log_beta = math.log(beta_mean) + beta_sigma * z
            self.beta = math.exp(log_beta)
        else:
            self.beta = beta_mean

        self.happy = False

    @property
    def neighbourhood(self):
        return self.model.cell_to_neighbourhood[self.cell.coordinate]

    def utility(self, neighbourhood, is_current: bool) -> float:
        """
        Cost-benefit utility of one neighbourhood for this agent.

        Affordability is a burden ratio: neighbourhood price divided by income.
        An agent is comfortable spending up to budget_fraction of its income
        (e.g. 50%) on housing. If a neighbourhood costs more than the agent's
        budget, it gets a penalty, and the penalty grows with how far over
        budget it is.
        """
        price = neighbourhood.cost
        burden = price / self.income
        excess = burden - self.budget_fraction

        if excess > 0:
            penalty = excess
        else:
            penalty = 0.0

        total_utility = self.baseline_benefit + self.quality_weight * neighbourhood.quality
        total_utility -= self.beta * self.logit_scale * penalty

        if not is_current:
            total_utility -= self.move_cost
        return total_utility

    def step(self) -> None:
        """
        Make a conditional choice over the available neighbourhoods.
        """
        current_neighbourhood = self.neighbourhood

        # Choose between the current neighbourhood and every neighbourhood with a vacancy.
        choice_set = [current_neighbourhood]
        for neighbourhood in self.model.neighbourhoods.values():
            if neighbourhood.id == current_neighbourhood.id:
                continue
            for cell in neighbourhood.cells:
                if cell.is_empty:
                    choice_set.append(neighbourhood)
                    break

        # Score each neighbourhood in the choice set.
        utilities = []
        for neighbourhood in choice_set:
            if neighbourhood.id == current_neighbourhood.id:
                is_current = True
            else:
                is_current = False
            utilities.append(self.utility(neighbourhood, is_current))

        max_utility = max(utilities)
        choice_weights = []
        for utility in utilities:
            choice_weights.append(math.exp(utility - max_utility))

        # Randomly pick one neighbourhood, weighted by the scores.
        chosen_neighbourhood = self.model.random.choices(choice_set, weights=choice_weights)[0]

        # If the chosen neighbourhood is the current neighbourhood, the agent is happy.
        if chosen_neighbourhood.id == current_neighbourhood.id:
            self.happy = True
            return

        # Otherwise move into one of the empty cells of the chosen neighbourhood.
        empty_cells = []
        for cell in chosen_neighbourhood.cells:
            if cell.is_empty:
                empty_cells.append(cell)

        if empty_cells:
            self.cell = self.model.random.choice(empty_cells)
            self.happy = False
        else:
            # The chosen neighbourhood is full, the agent is unhappy.
            self.happy = False

    def assign_state(self) -> None:
        """
        Count this agent as happy if its logit draw left it in place this step.
        Model.step resets the counter to 0 before re-tallying each step.
        """
        if self.happy:
            self.model.happy += 1


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
