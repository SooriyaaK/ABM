#
# This code is a rough outline currently, since not all classes are properly defined yet
# So: DO NOT ATTEMPT TO RUN THIS
#

import numpy as np
from typing import List

def rank_agents_percentile(agents: List[SchellingAgent]) -> None:
    """Ranks agents on percentiles between low and high"""

    #TODO: somehow get agent's income !!
    incomes = np.array([a.income for a in agents])
    T = len(agents)

    # argsort twice: positions -> ranks
    order = np.argsort(incomes)
    ranks = np.empty(T, dtype=float)
    ranks[order] = np.arange(T) / T  # or /(T-1) if you prefer 0..1 inclusive

    for agent, r in zip(agents, ranks):
        agent.rank = r


def create_thresholds(n:int=99):
    """Creates n percentile thresholds between 0.01 and 0.99 in rank space."""
    thresholds = np.linspace(0.01, 0.99, n)
    return thresholds

def f_global_entropy_split(threshold:float) -> float:
    """compute global entrop split for certain threshold"""
    E_g = -(threshold * np.log2(threshold) + (1-threshold) * np.log2(1-threshold))
    return E_g

def f_local_entropy_split(threshold:float, neighbourhood) -> float:
    # TODO: how is a neighbourhood class defined?
    """compute local entropy split for certain threshold"""

    ranks = np.array([a.rank for a in neighbourhood.agents]) # NOTE problem with empty neighbourhoods
    pnk = np.count_nonzero(ranks <= threshold) / len(ranks) # fraction of agents with income below thershold in current neighbourhood; had to be an np.array I think
    
    if pnk == 0 or pnk == 1: #avoiding log(0)
        E_l = 0
    else:
        E_l = -(pnk * np.log2(pnk) + (1-pnk) *np.log2(1-pnk))

    return E_l


def f_Hk(threshold:float, neighbourhood: Neighbourhood, tot_agents:int, Eg:float) -> float:
    """H_k formula provided by Reardon (2011) (one value of the pairwise information theory index)"""
    El = f_local_entropy_split(threshold, neighbourhood)
    tn = len(neighbourhood.agents) # or smth like this; NOTE probem with empty neighbourhoods

    if tn == 0 or Eg == 0.0: # avoid problems with empty neighbourhoods
        return 0.0

    # for neighbourhood i, get information theory index
    H_ki = (tn/(tot_agents * Eg)) * (Eg - El)
    return H_ki


def compute_HR(neighbourhoods: List[Neighbourhood], agents: List[SchellingAgent], n_threshold=99):
    """ Does numerical integration to retrieve H_R value for full simulation at current timestep""" 
    # 1. rank agents
    rank_agents_percentile(agents)
    T = len(agents)

    # 2. define thresholds
    thresholds = create_thresholds(n_threshold)

    # 3. looping over thresholds, compute Hk for each neighbourhood
    H_values = []
    for p_k in thresholds:
        Eg = f_global_entropy_split(p_k)
        Hk = 0.0
        for neighbourhood in neighbourhoods:
            Hk += f_Hk(p_k, neighbourhood, T, Eg)
        H_values.append(Hk)

    # 4. integrate using trapezoid rule
    H_values = np.array(H_values)
    E_values = -thresholds * np.log2(thresholds) - (1 - thresholds) * np.log2(1 - thresholds)
    H_R = 2 * np.log(2) * np.trapz(E_values * H_values, thresholds)
    return H_R





