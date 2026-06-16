Here's a cleaner and more polished version of your text:

# ABM – Agent-Based Modelling

### Neighborhood Initialization

* How should the neighborhood be initialized at the beginning of the simulation?

* Residents are randomly distributed across neighborhoods.

* Neighborhoods may expand over time based on population density and the income levels of their residents.

### Agent Decision-Making

* Each person evaluates whether the average income level of their current neighborhood exceeds what they can afford.

* If the neighborhood is too expensive, they decide to relocate.

* The person moves to a more affordable neighborhood that better matches their income level.

### Cost–Benefit Analysis
* Before relocating, individuals perform a cost–benefit analysis to determine whether moving is worthwhile.

* Costs may include moving expenses, commuting costs, housing costs, and social disruption.

* Benefits may include improved living conditions, access to amenities, shorter commute times, and greater economic opportunities.

* An individual relocates only if the expected benefits outweigh the associated costs.

### Rational Decision-Making

* Assume each agent behaves rationally and seeks to maximize its utility.

*An agent's utility function could include:

[
U_i = \text{Neighborhood Quality} + \text{Amenities} - \text{Housing Cost} - \text{Commuting Cost} - \text{Moving Cost}
]

* Each individual chooses the neighborhood that maximizes their utility subject to affordability constraints.

* If the cost of living exceeds an agent's income threshold, the utility of remaining decreases.

* The agent evaluates alternative neighborhoods and moves to the one with the highest utility.

### Nash Equilibrium

* A Nash equilibrium occurs when no agent can improve their utility by unilaterally changing neighborhoods, given the choices of all other agents.

* Agents move between neighborhoods based on their preferences and constraints.

* Moving changes neighborhood composition, which affects housing costs and attractiveness.

* Eventually, the system may reach a stable state where no agent wants to relocate.

That stable configuration can be interpreted as a Nash equilibrium.

* The equilibrium may not be socially optimal.
* Multiple equilibria may exist.
* The system may never converge if the environment changes continuously.


### Summary: Lecture 4

Housing costs depend on population density. As more agents move into a neighborhood, demand for housing increases, leading to higher housing expenses.
This creates a dynamic feedback loop:

1. Agents choose neighborhoods that maximize their utility.
2. Popular neighborhoods attract more residents.
3. Increased population density raises housing costs.
4. Some agents can no longer afford to remain in the neighborhood.
5. Agents relocate to less dense and more affordable areas.
6. The composition and density of neighborhoods evolve over time.

These relocation decisions can be modeled using discrete choice models, where agents evaluate trade-offs between housing costs, neighborhood quality, accessibility, and travel time. Such local interactions may lead to emergent patterns such as urban expansion, segregation, and gentrification.


