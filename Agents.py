from mesa.discrete_space import CellAgent


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
        self.homophily = homophily
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
        if self.happy: 
            return
        if self.type == 3:
            higher_nb = [nb for nb in self.model.neighbourhoods.values() if nb.cost > self.neighbourhood.cost]
            candidate_nbs = [c for n in higher_nb for c in n.cells if c.is_empty]
            if candidate_nbs:
                self.cell = self.model.random.choice(candidate_nbs)
        else:
            lower_nb = [nb for nb in self.model.neighbourhoods.values() if nb.cost < self.neighbourhood.cost]
            candidate_nbs = [c for n in lower_nb for c in n.cells if c.is_empty]
            if candidate_nbs:
                self.cell = self.model.random.choice(candidate_nbs)
            #affordable = [nb for nb in self.model.neighbourhoods.values() if nb.cost <= self.income]
            #empty = [c for n in affordable for c in n.cells if c.is_empty]
            #if empty:
            #    self.cell = self.model.random.choice(empty)

