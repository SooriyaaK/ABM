from mesa.discrete_space import CellAgent
import random


class SchellingAgent(CellAgent):
    """Schelling segregation agent."""

    def __init__(
        self, model, cell, agent_type: int, income: int, homophily: float = 0.4, radius: int = 1
    ) -> None:
        """Create a new Schelling agent.
        Args:
            model: The model instance the agent belongs to
            agent_type: Indicator for the agent's type (minority=1, majority=0)
            homophily: Minimum number of similar neighbors needed for happiness
            radius: Search radius for checking neighbor similarity
        """
        super().__init__(model)
        self.cell = cell
        self.type = agent_type
        # self.homophily = homophily

        self.cost_weight = max(0, min(1, random.gauss(0.5, 0.2)))  # random (gaussian) float between [0,1]
        self.homophily_weight = 1 - self.cost_weight  # weights sum to 1

        self.radius = radius
        self.happy = False
        self.income = income

    @property
    def neighbourhood(self):
        return self.model.cell_to_neighbourhood[self.cell.coordinate]

    def assign_state(self) -> None:
        """Determine if agent is happy and move if necessary."""
        neighbors = list(self.cell.get_neighborhood(radius=self.radius).agents)

        # Count similar neighbors
        similar_neighbors = len([n for n in neighbors if n.type == self.type])
        costs = {nb.id: nb.cost for nb in self.model.neighbourhoods.values()}

        # Calculate the fraction of similar neighbors
        if (valid_neighbors := len(neighbors)) > 0:
            similarity_fraction = similar_neighbors / valid_neighbors
        else:
            # If there are no neighbors, the similarity fraction is 0
            similarity_fraction = 0.0
        if self.type == 3:
            highest_cost = max(n.cost for n in self.model.neighbourhoods.values())
            higher_exists = highest_cost > self.neighbourhood.cost
            if higher_exists and self.model.random.random() < 0.3:
                self.happy = False
            else:
                self.happy = True
                self.model.happy += 1
            #self.happy = self.neighbourhood.cost >= highest_cost
        else:
            if self.neighbourhood.cost <= self.income:
                self.happy = True
                self.model.happy += 1
            else:
                self.happy = False
        #if similarity_fraction < self.homophily:
            #self.happy = False
        #else:
            #self.happy = True
            #self.model.happy += 1

    def step(self) -> None:
        # Move if unhappy
        # if self.happy: 
        #     return
        # if self.type == 3:
        #     higher_nb = [nb for nb in self.model.neighbourhoods.values() if nb.cost > self.neighbourhood.cost]
        #     candidate_nbs = [c for n in higher_nb for c in n.cells if c.is_empty]
        #     if candidate_nbs:
        #         self.cell = self.model.random.choice(candidate_nbs)
        # else:
        #     lower_nb = [nb for nb in self.model.neighbourhoods.values() if nb.cost < self.neighbourhood.cost]
        #     candidate_nbs = [c for n in lower_nb for c in n.cells if c.is_empty]
        #     if candidate_nbs:
        #         self.cell = self.model.random.choice(candidate_nbs)
            #affordable = [nb for nb in self.model.neighbourhoods.values() if nb.cost <= self.income]
            #empty = [c for n in affordable for c in n.cells if c.is_empty]
            #if empty:
            #    self.cell = self.model.random.choice(empty)

        max_income = 4000 #TODO: this is now hardcoded; bad idea
        current_nb = self.model.cell_to_neighbourhood[self.cell.coordinate]

        def utility(nb):
            """Calculates utility of a macro-neighbourhood"""
            agents = nb.agents
            if agents:
                homophily_score = sum(1 for a in agents if a.type == self.type) / len(agents)
            else:
                homophily_score = 1.0  # empty neighbourhood -> no dissimilar neighbours
            cost_score = max(0.0, 1 - abs(nb.cost - self.income) / max_income)
            epsilon = self.model.random.gauss(0, 1)  # noise factor from gumbel distr.
            return self.homophily_weight * homophily_score + self.cost_weight * cost_score + epsilon # basic RUM func

        U_stay = utility(current_nb) # utility of current neighbourhood

        # Find best candidate neighbourhood (excluding current; must have empty cell)
        best_nb = None
        best_utility = U_stay  # only move if better than staying

        for nb in self.model.neighbourhoods.values():
            # loop over all neighbourhoods
            if nb is current_nb:
                # skip current neighbourhood
                continue
            if not any(c.is_empty for c in nb.cells):
                # skip all neighbourhoods without empty cells since we can't move ther anyway
                continue
            U = utility(nb) # calculate utility of neighbourhood
            if U > best_utility:
                best_utility = U
                best_nb = nb

        # Move to a random empty cell in the best neighbourhood
        if best_nb is not None:
            empty_cells = [c for c in best_nb.cells if c.is_empty]
            self.cell = self.model.random.choice(empty_cells)
        



