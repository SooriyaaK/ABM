#
# This code is a rough outline currently, since not all classes are properly defined yet
# So: DO NOT ATTEMPT TO RUN THIS
#

import numpy as np
from typing import List

def rank_agents_percentile(agents: List[SchellingAgent]):
    """Ranks agents on percentiles between low and high"""

    #TODO: somehow get agent's income !!
    T = len(agents)
    ranks = agents.argsort().argsort() / T  # percentile ranks in [0, 1); TODO: rank on income!!
    return ranks # return sorted agents


def create_thresholds(n:int=99):
    """Creates n income threshold values between 0.01 and 0.99"""
    thresholds = np.linspace(0.01, 0.99, n)
    return thresholds

def f_global_entropy_split(threshold:float) -> float:
    """compute global entrop split for certain threshold"""
    E_g = -(threshold * np.log2(threshold) + (1-threshold) * np.log2(1-threshold))
    return E_g

def f_local_entropy_split(threshold:float, neighbourhood) -> float:
    # TODO: how is a neighbourhood class defined?
    """compute local entropy split for certain threshold"""
    agents = neighbourhood.agents() #TODO: this is fake code as neighbourhood class is not yet defined
    pnk = numpy.count_nonzero(np.array(agents.income()) < threshold) / len(agents) # fraction of agents with income below thershold in current neighbourhood; had to be an np.array I think
    
    if pnk == 0 or pnk == 1: #avoiding log(0)
        E_l = 0
    else:
        E_l = -(pnk * np.log2(pnk) + (1-pnk) *np.log2(1-pnk))
        
    return E_l


def f_Hk(threshold:float, neighbourhood: Neighbourhood, tot_agents:int, Eg:float) -> float:
    """H_k formula provided by Reardon (2011) (one value of the pairwise information theory index)"""
    El = f_local_entropy_split(threshold, neighbourhood)
    # Eg = f_global_entropy_split(threshold)
    tn = len(neighbourhood.agents()) # or smth like this
    T = tot_agents # total number of agents in this simulation

    # for neighbourhood i, get information theory index
    H_ki = (tn/(T * Eg)) * (Eg - El)
    return H_ki


def compute_HR(neighbourhoods: list[Neighbourhood], agents: list[SchellingAgent], n_threshold=99):
    """ Does numerical integration to retrieve H_R value for full simulation at current timestep""" 
    # 1. rank agents
    agents = rank_agents_percentile(agents)

    # 2. define thresholds
    thresholds = create_thresholds(n_threshold)

    # 3. looping over thresholds, compute Hk for each neighbourhood
    H_values = []
    for p_k in thresholds:
        Eg = f_global_entropy_split(p_k)
        Hk = 0.0
        for neighbourhood in neighbourhoods:
            Hk += f_Hk(p_k, neighbourhood, len(agents), Eg)
        H_values.append(Hk)

    # 4. integrate using trapezoid rule
    H_values = np.array(H_values)
    E_values = -thresholds * np.log2(thresholds) - (1 - thresholds) * np.log2(1 - thresholds)
    H_R = 2 * np.log(2) * np.trapz(E_values * H_values, thresholds)
    return H_R



    











