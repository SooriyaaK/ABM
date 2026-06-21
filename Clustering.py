import numpy as np

def compute_morans_I(model):
    """
    Compute Moran's I for income clustering across the grid.
    This is fine-grained cell-level clustering.
    
    Returns a value between -1 and 1:
      +1 = perfect clustering (similar incomes next to each other)
       0 = random spatial distribution
      -1 = perfect dispersion (checkerboard pattern)
    """
    # Collect all occupied cells with their income and coordinates
    cells_data = []
    for agent in model.agents:
        x, y = agent.cell.coordinate
        cells_data.append((x, y, agent.income))

    if len(cells_data) < 2:
        return None

    coords = [(x, y) for x, y, _ in cells_data]
    incomes = np.array([inc for _, _, inc in cells_data], dtype=float)
    n = len(incomes)
    mean_income = incomes.mean()
    deviations = incomes - mean_income

    # Build a lookup for quick coordinate → deviation access
    coord_to_dev = {(x, y): dev for (x, y), dev in zip(coords, deviations)}

    # Moore neighborhood offsets (8 neighbors)
    offsets = [(-1,-1),(-1,0),(-1,1),
               ( 0,-1),        (0,1),
               ( 1,-1),( 1,0),( 1,1)]

    W = 0.0        # total weight (number of valid neighbor pairs)
    spatial_sum = 0.0

    for (x, y), dev_i in zip(coords, deviations):
        for dx, dy in offsets:
            neighbor = (x + dx, y + dy)
            if neighbor in coord_to_dev:
                spatial_sum += dev_i * coord_to_dev[neighbor]
                W += 1.0

    if W == 0:
        return None

    denominator = np.sum(deviations ** 2)
    if denominator == 0:
        return None

    I = (n / W) * (spatial_sum / denominator)
    return float(I)


# Maximum possible between-neighbourhood variance
# occurs when 1/3 of neighbourhoods have avg income 1000, 1/3 have 2000, 1/3 have 4000
INCOME_LEVELS = [1000, 2000, 4000]
GLOBAL_MEAN = np.mean(INCOME_LEVELS)  # 2333
MAX_BETWEEN_VAR = np.var(INCOME_LEVELS)  # ≈ 1,400,000



def compute_neighbourhood_income_variance(model):
    """
    Compute the variance of average incomes across neighbourhoods.
    This is macro-level sorting between neighbourhoods (between-neighbourhood differences in average income).
    It answers the question: have distinct rich and poor neighbourhoods formed?
    We normalise to between 0 and 1.
    
    High variance = neighbourhoods are very different from each other income-wise
    (strong between-neighbourhood segregation).
    Low variance = neighbourhoods are similar to each other.
    """
    avg_incomes = []
    for nb in model.neighbourhoods.values():
        if nb.num_agents > 0:
            avg_income = sum(a.income for a in nb.agents) / nb.num_agents
            avg_incomes.append(avg_income)

    if len(avg_incomes) < 2:
        return None

    raw = float(np.var(avg_incomes))
    return raw / MAX_BETWEEN_VAR   # NORMALISED: now 0 = fully mixed, 1 = fully segregated