import random
from mesa.discrete_space import CellAgent
class SchellingAgent(CellAgent):
    """Schelling segregation agent."""

    def __init__(
        self, model, cell, agent_type: int, homophily: float = 0.4, radius: int = 1
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
        self.income_by_type = {
            1: 1500,
            2: 3000,
            3: 4500
        }
        self.income = self.income_by_type[self.type]

        self.move_to(cell)
        self.moving_cost = 5.0
        self.utility = 0.0

    def can_afford(self, cell):
        """
        Return True if this agent can afford the given cell.
        """
        return self.income > cell.housing_cost

    def assign_state(self) -> None:
        """Determine if agent is happy and move if necessary."""
        neighbors = list(self.cell.get_neighborhood(radius=self.radius).agents)

        # Count similar neighbors
        similar_neighbors = len([n for n in neighbors if n.type == self.type])

        # Calculate the fraction of similar neighbors
        if (valid_neighbors := len(neighbors)) > 0:
            similarity_fraction = similar_neighbors / valid_neighbors
        else:
            # If there are no neighbors, the similarity fraction is 0
            similarity_fraction = 0.0

        if similarity_fraction < self.homophily:
            self.happy = False
        else:
            self.happy = True
            self.model.happy += 1

    def step(self) -> None:
        # Move if unhappy
        if not self.happy:
            affordable_cells = []

            for cell in self.model.grid.empties.cells:
                if self.can_afford(cell):
                    affordable_cells.append(cell)


            if affordable_cells:
                    new_cell = random.choice(affordable_cells)
                    self.move_to(new_cell)
                    
            # self.cell = self.model.grid.select_random_empty_cell()

    
    def calculate_utility(self, cell, neighborhood, alpha, beta) -> float:
        """
        Calculate the utility of a given cell based on the similarity of neighbors.

        BENEFITS:
        - Living near similar people?
        - Good neighborhood quality?
        - Affordable housing?
        - Close to job?
        
        COSTS:
        - Housing expense
        - Commute time
        - Distance from community
        
        CONSTRAINTS:
        - Must be able to afford?
        - Minimum similarity required?
        
        RETURN: float: utility score based on benefits, costs, and constraints.
        """
        neighbors = list(self.cell.get_neighborhood(radius=self.radius).agents)
        total_neighbours = len(neighbors)
        # Constraints
        
        if self.income < self.cell.housing_cost:
            return 0.0  # Cannot afford the housing cost
        
        similar_neighbours = 0 

        for neighbor in neighbors:
            if neighbor.type == self.type:
                similar_neighbours += 1
        
        # Calculate Living near similar people
        if total_neighbours > 0:
            social_utility = similar_neighbours / total_neighbours
        else:
            social_utility = 0.5 

        # Calculate economic utility
        affordability = (self.income - cell.housing_cost) / self.income

        utility = (alpha * social_utility) + (beta * affordability)

        return utility

        