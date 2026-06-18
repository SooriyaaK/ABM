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

$U_i = \text{Neighborhood Quality} + \text{Amenities} - \text{Housing Cost} - \text{Commuting Cost} - \text{Moving Cost}$

* Each individual chooses the neighborhood that maximizes their utility subject to affordability constraints.

* If the cost of living exceeds an agent's income threshold, the utility of remaining decreases.

* The agent evaluates alternative neighborhoods and moves to the one with the highest utility.

### Mixed logit

The model uses a mixed logit: every agent draws its own price-sensitivity once from a lognormal distribution, so two agents with the same income can still react differently to the same prices. This price-sensitivity controls how strongly an agent reacts when a neighbourhood costs more than it can comfortably afford: a high value means the agent strongly avoids overpriced areas, while a low value means it tolerates them. Drawing it from a lognormal distribution keeps the value positive and gives a realistic spread, with most agents near a typical sensitivity and a few much more reactive. The spread of this distribution is a single dial: set it to zero and everyone behaves identically, turn it up and the population becomes more varied.

### Utility

* Before deciding where to live, an agent gives every neighbourhood it could move to a utility score, and the score captures the trade-off between what a place offers and what it costs.
* The score goes up with the neighbourhood's quality — how nice or desirable the area is — and goes down by a moving cost whenever the agent would have to relocate.
* It also goes down when the neighbourhood is unaffordable: as long as housing stays within what the agent can comfortably spend there is no penalty, but once it costs more the penalty kicks in and grows the further over budget it gets.
* The agent then leans toward the neighbourhood with the highest score, which is how richer agents end up in the desirable, expensive areas and poorer agents are pushed toward the cheaper ones.

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

---

### Model Converge
In order to determine when the model converges, we need some metrics that measure the (macro) neighbourhoods' stability.
* EXAMPLE: for each neighbourhood, measure average income level. Define model stability as the average neighborhood income of all neighbourhoods to not have changed more than some threshold $\epsilon$ for some number of $n$ iterations, i.e
    * $\max_{T-n < t \leq T} \;\max_{k} \left|\mu_t(k) - \mu_{t-1}(k)\right| < \epsilon$

### `convergence.py`

This module implements the **income segregation metric** and tools needed to detect convergence in the Schelling model.

#### What it measures

We compute **Reardon’s rank‑order information theory index** $H^R$ (https://cepa.stanford.edu/sites/default/files/reardon%20measures%20of%20income%20segregation%20sept2011.pdf), which measures how strongly agents are spatially sorted by **income rank** across neighborhoods:

- $H^R = 0$ → no income segregation (every neighborhood has the same income distribution as the whole population).
- $H^R \approx 1$ → extreme segregation (neighborhoods are almost perfectly stratified by income).

We use $H^R$ at each iteration of the simulation to track how the segregation pattern evolves and to decide when the system has converged.

---

#### Core steps in `convergence.py`

Given a list of `SchellingAgent` objects and `Neighbourhood` objects, the module does:

1. **Rank agents by income (global percentiles)**  
   - `rank_agents_percentile(agents)`  
   - Extracts each agent’s `.income`, sorts all agents, and assigns each one a percentile rank `agent.rank` in $[0,1)$ (e.g. 0.0 = poorest, 1.0 = richest).

2. **Define percentile thresholds**  
   - `create_thresholds(n=99)`  
   - Creates `n` equally spaced percentile thresholds between 0.01 and 0.99 in **rank space** (not raw income).  
   - These thresholds are used to dichotomize the population into “below threshold” vs “above threshold” groups at many points along the income distribution.

3. **Compute the pairwise information index at each threshold**  
   For each threshold $p_k$:
   - Compute the **global entropy** of the split using the fraction $p_k$ of agents with rank ≤ $p_k$:
     - `f_global_entropy_split(p_k)`  
     - This is $E(p_k) = -[p_k \log_2 p_k + (1-p_k)\log_2(1-p_k)]$.
   - For each `Neighbourhood`, compute the **local entropy**:
     - `f_local_entropy_split(p_k, neighbourhood)`  
     - Uses the fraction of agents in that neighborhood whose `agent.rank <= p_k`.
   - Combine these to get that neighborhood’s contribution to the pairwise information index at $p_k$:
     - `f_Hk(p_k, neighbourhood, T, Eg)`  
     - Implements Reardon’s $$H_k$$ formula and returns the contribution of one neighborhood.
   - Sum over all neighborhoods to obtain the overall $H_k$ at threshold $p_k$.

4. **Integrate over thresholds to get $$H^R$$**  
   - `compute_HR(neighbourhoods, agents, n_threshold=99)`  
   - Runs steps (1)–(3), building up the sequence $\{H_k\}$ over all thresholds.
   - Computes the entropy weights $E(p_k)$ at each threshold.
   - Uses the **trapezoidal rule** (`numpy.trapz`) to numerically approximate
     $
     H^R \approx 2 \ln(2) \int_0^1 E(p)\,H(p)\,dp
     $
   - Returns a single scalar $H^R$ for the current iteration.

---

#### How it is used in the simulation

In the main model loop (not in this file), you typically:

1. Run one Schelling update step (agents move).
2. Call `compute_HR(neighbourhoods, agents, ...)` to get the current $H^R$.
3. Append the value to a history list, e.g. `H_history`.
4. Check a **convergence criterion**, for example:
   - The absolute change in $H^R$ is below some threshold $\epsilon$ for several consecutive iterations:
     $
     |H^R_t - H^R_{t-1}| < \epsilon
     $
   - If this holds (and/or a max number of iterations is reached), we declare the system converged and stop the run.

---

#### Assumptions about `Agent` and `Neighbourhood`

The functions in `convergence.py` assume:

- `SchellingAgent` has at least:
  - `income: float` — static or slowly changing attribute.
  - `rank: float` — set by `rank_agents_percentile` each iteration (global percentile in $[0,1)$).
- `Neighbourhood` has:
  - `agents: list[SchellingAgent]` — all agents currently living in that neighborhood.

No additional methods are required from these classes for the convergence code to work.
