#
# This code is a rough outline currently, since not all classes are properly defined yet
# So: DO NOT ATTEMPT TO RUN THIS
#
# turn off calc_second_order in SALib
# 
# The Multigroup Entropy Index (Also Known as Theil's H or the Information Theory Index)
# https://www.researchgate.net/publication/266452850_The_Multigroup_Entropy_Index_Also_Known_as_Theil's_H_or_the_Information_Theory_Index
#

import numpy as np
from typing import List


def f_overall_entropy(agents) -> float:
    """Compute global entropy E_T over income groups, all agents."""
    incomes = np.array([a.income for a in agents])
    if len(incomes) == 0: # no agents -> return zero
        return 0.0

    # counts per income group
    values, counts = np.unique(incomes, return_counts=True)
    p_g = counts / counts.sum()  # fraction for each group

    # avoid log2(0): mask sum for p_g > 0
    mask = p_g > 0
    p_g = p_g[mask]

    Et = -np.sum(p_g * np.log2(p_g))
    return Et


def f_neighbourhood_entropy(neighbourhood) -> float:
    """Compute neighbourhood entropy E_n over income groups, only in this neighbourhood."""
    agents = neighbourhood.agents
    incomes = np.array([a.income for a in agents])
    if len(incomes) == 0: # no agents -> return zero
        return 0.0

    values, counts = np.unique(incomes, return_counts=True)
    p_ng = counts / counts.sum() # fraction for each group

    mask = p_ng > 0 # mask zeroes
    p_ng = p_ng[mask]

    En = -np.sum(p_ng * np.log2(p_ng))
    return En


def compute_H(neighbourhoods, agents) -> float:
    Et = f_overall_entropy(agents)
    if Et == 0.0: # no agents
        return 0.0

    T = len(agents) # total number of agents
    H = 0.0
    for n in neighbourhoods:
        tn = len(n.agents)
        if tn == 0: # no agents
            continue
        En = f_neighbourhood_entropy(n)
        H += ( (tn / T) * (Et - En) ) / Et
    return H





