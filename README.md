# ABM – Agent-Based Modelling
This github repo consists on an extension of the classic Schelling segregation model, meant to simulate gentrification and its consequential displacement as an emergent property of income groups. Our research question is: to what extent do agent density, cooperative
strategy distribution, neighbourhood size, and mobility rate influence the degree of segregation
arising from gentrification and displacement processes?

# Setup Instructions

To be able to run anything that is given in this project a uv environment needs to be created.

### 1. Install `uv`

To make the management of dependencies easier it is recommended that you use `uv`. If you already have `uv` installed this step can be skipped. 

##### For Mac / Linux users open a terminal and run
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

##### For Windows users open PowerShell and run
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After the installation is finished, quit and reopen your terminal/powershell.

### 2. Install the necessary dependencies and Python
Run the below command on the same directory as this repository
```bash
uv sync
```
This will install all necessary packages and create the `uv` environment.

After all these steps your environment will be ready with all the packages. After these steps if you want to run a visualized version of the model you can run the command:
```bash
uv run solara run App.py
```

----

### Mixed logit

The model uses a mixed logit: every agent draws its own price-sensitivity once from a lognormal distribution, so two agents with the same income can still react differently to the same prices. This price-sensitivity controls how strongly an agent reacts when a neighbourhood costs more than it can comfortably afford: a high value means the agent strongly avoids overpriced areas, while a low value means it tolerates them. Drawing it from a lognormal distribution keeps the value positive and gives a realistic spread, with most agents near a typical sensitivity and a few much more reactive. The spread of this distribution is a single dial: set it to zero and everyone behaves identically, turn it up and the population becomes more varied.

### Utility

* Before deciding where to live, an agent gives every neighbourhood it could move to a utility score, and the score captures the trade-off between what a place offers and what it costs.
* The score goes up with the neighbourhood's quality — how nice or desirable the area is — and goes down by a moving cost whenever the agent would have to relocate.
* It also goes down when the neighbourhood is unaffordable: as long as housing stays within what the agent can comfortably spend there is no penalty, but once it costs more the penalty kicks in and grows the further over budget it gets.
* The agent then leans toward the neighbourhood with the highest score, which is how richer agents end up in the desirable, expensive areas and poorer agents are pushed toward the cheaper ones.


### Dynamic Feedback Loop

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

## Model Convergence

### `convergence.py`

This module implements the **income segregation metric** and the tools needed to detect convergence in the Schelling model.

---

#### What it measures

We compute the **Multigroup Entropy Index** $H$ (https://www.researchgate.net/publication/266452850_The_Multigroup_Entropy_Index_Also_Known_as_Theil's_H_or_the_Information_Theory_Index), which measures how strongly agents are spatially sorted by **income group** across neighbourhoods:

- $H = 0$ → no income segregation (every neighbourhood has the same income-group distribution as the whole population).
- $H = 1$ → complete segregation (each neighbourhood contains only one income group).

$H$ is computed at each time step to track how the segregation pattern evolves and to decide when the system has converged.

> **Note:** $H$ is computed over the three **discrete income groups** (low = 1000, middle = 2000, high = 4000), not over a continuous income rank or percentile.

---

#### Core steps in `convergence.py`

Given a list of `SchellingAgent` objects and `Neighbourhood` objects, the module:

1. **Computes global entropy $E_T$** (`f_overall_entropy`) — the Shannon entropy of the income-group distribution across *all* agents:

$$E_T = -\sum_{g} p_g \log_2 p_g$$

where $p_g$ is the fraction of all agents belonging to income group $g \in \{1000, 2000, 4000\}$.

2. **Computes neighbourhood entropy $E_n$** (`f_neighbourhood_entropy`) — the same Shannon entropy, but computed only over the agents residing in neighbourhood $n$:

$$E_n = -\sum_{g} p_{ng} \log_2 p_{ng}$$

where $p_{ng}$ is the fraction of agents in neighbourhood $n$ belonging to group $g$.

3. **Computes the Multigroup Entropy Index $H$** (`compute_H`) — a weighted average of how much each neighbourhood's entropy deviates from the global entropy:

$$H = \sum_{n} \frac{t_n}{T} \cdot \frac{E_T - E_n}{E_T}$$

where $t_n$ is the number of agents in neighbourhood $n$ and $T$ is the total number of agents. Empty neighbourhoods are skipped. If $E_T = 0$ (all agents belong to the same income group), $H$ is defined as $0$.

---

#### How it is used in the simulation

In `Model.py`, `compute_H` is called at the end of every time step:

```python
H = compute_H(list(self.neighbourhoods.values()), list(self.agents))
self.H_history.append(H)
```

The model then checks a **convergence criterion**. The simulation is declared converged and halted early if *all* of the following hold:

- At least `min_step_count = 150` steps have been completed.
- The range of $H$ over the last 50 steps is below $\varepsilon = 0.01$:

$$\max(H_{t-49},\ \ldots,\ H_t) - \min(H_{t-49},\ \ldots,\ H_t) < \varepsilon$$

If this criterion is not met, the simulation continues until `max_steps = 500` is reached.

---

#### Assumptions about `SchellingAgent` and `Neighbourhood`

The functions in `convergence.py` assume:

- `SchellingAgent` has:
  - `income: int` — one of `{1000, 2000, 4000}`, used to identify the agent's income group. This attribute is **static** (does not change during the simulation).
- `Neighbourhood` has:
  - `agents` — a **property** returning a list of all `SchellingAgent` objects currently residing in that neighbourhood (assembled by iterating over its cells).

No additional methods or attributes are required from these classes for the convergence code to work.


---

# Sobol Sensitivity Analysis
 
We perform variance-based global sensitivity analysis using first-order (S1) and total-order (ST) Sobol indices , implemented via the SALib Python library. This quantifies how much of the variance in each output metric (segregation index H, Moran's I, between-neighbourhood income variance) is attributable to each of the four model parameters: `density`, `defector_frac`, `neighbourhood_count`, and `activation_rate`.
 
## Method
 
Rather than a factorial grid, we use the **Saltelli sampling scheme**, which constructs two independent quasi-random matrices A and B of shape N × k, then builds k additional matrices A_B^i by swapping column i of A with column i of B. This yields N × (k + 2) total parameter combinations — in our case 1024 × 6 = **6144 model runs**. Each run uses 10 random seeds to average out stochastic noise, giving a signal-to-noise ratio of ~4.5.
 
- **S1 (first-order)**: fraction of output variance explained by parameter i alone
- **ST (total-order)**: fraction explained by parameter i including all interactions with other parameters
- **ST − S1**: interaction effect of parameter i with other parameters
N=1024 was chosen as a power of 2 (2^10), which is required for the convergence properties of the Sobol' quasi-random sequence.

 
## Pipeline
 
Follow these steps in order to reproduce the full sensitivity analysis.
 
### 0. Install dependencies
```bash
uv add SALib numpy matplotlib scipy pandas
```
 
### 1. Generate the Saltelli sample
```bash
uv run python Saltelli.py
```
Creates `results_saltelli/saltelli_params.json` (6144 combos), `saltelli_problem.json`, and `saltelli_problem_X.npy`. Only run this once — regenerating creates a new random matrix that no longer matches any existing simulation results.
 
### 2. (Optional) Run a quick test before the full job
```bash
# generate 5 test combos
uv run python Params_test.py
 
# submit test array (5 jobs, 3 seeds, 15 min)
mkdir -p results_saltelli
sbatch run_job_saltelli_test.sh
 
# check logs
cat results_saltelli/schelling_test_*.out
 
# check output files
ls results_saltelli/run_*.npz | wc -l  # should be 5
```
 
### 3. Submit the full SLURM array
```bash
sbatch run_job_saltelli.sh
```
Submits 6144 jobs throttled to 200 at a time. Expected wall time ~1.5 hours.
 
### 4. Monitor progress
```bash
# how many jobs still running
squeue -u $USER
 
# how many output files created so far
ls results_saltelli/run_*.npz | wc -l  # done when this hits 6144
 
# check for failed jobs
sacct -j <JOBID> --format=JobID,State | grep FAILED | grep -v "+"
```
 
### 5. Resubmit any failed jobs
```bash
# resubmit specific failed indices (e.g 66,68, and 70)
sbatch --array=66,68,70 run_job_saltelli.sh
```
 
### 6. Check signal-to-noise ratio
```bash
uv run python -c "
import numpy as np, glob
files = glob.glob('results_saltelli/run_*.npz')
within, across = [], []
for f in files:
    d = np.load(f, allow_pickle=True)
    tail = d['all_H'][:, -50:].mean(axis=1)
    within.append(tail.std())
    across.append(tail.mean())
print(f'Mean within-combo std: {np.mean(within):.4f}')
print(f'Std across combos:     {np.std(across):.4f}')
print(f'Signal/noise ratio:    {np.std(across)/np.mean(within):.1f}x')
"
```
We aim for a ratio around 3-5, which indicates that the parameter effects dominate seed noise and the Sobol indices are trustworthy.
 
### 7. Collect outputs
```bash
uv run python Collect_outputs.py
```
Collapses all `.npz` files to scalar Y vectors. Missing combos are filled with the mean. Should print `Collected 6144 combos`.
 
Verify:
```bash
ls results_saltelli/Y_*.npy
uv run python -c "
import numpy as np
Y = np.load('results_saltelli/Y_H.npy')
print(f'Length: {len(Y)}')       # should be 6144
print(f'Any NaN: {np.isnan(Y).sum()}')  # should be 0
"
```
Optional to generate heatmap plots: you can run the below command to create a csv file called `results_summary.csv` to get a summarized version of the npz files for easier plotting

```bash
uv run python Aggregate.py
```

### 8. Run the Sobol analysis
```bash
uv run python Analyse_sobol.py
```
Prints S1 and ST indices to console, saves `sobol_indices.png` and `sobol_indices.csv` into a directory it creates called `sobol_output`.

### 9. Run SensitivityAnalysis.py 
```bash
uv run python SensitivityAnalysis.py
```
Saves the H-convergence plots into the `sobol_output` directory.

### 10. Run Heatmap.py
```bash
uv run python Heatmap.py
```
Saves scatter plots and heatmaps into `sobol_output` directory. The parameters these plots are created with hardcoded as the parameters with the highest effects. 

## Files
 
| File | Description |
|---|---|
| `Saltelli.py` | Generates the Saltelli sample. Saves `saltelli_params.json` (parameter combos for SLURM), `saltelli_problem.json` (bounds and names for SALib), and `saltelli_problem_X.npy` (raw sample matrix needed for scatter plots). Run once before submitting jobs. |
| `run_job_saltelli.sh` | SLURM array job script. Submits 6144 jobs (`--array=0-6143%200`), each running `Run_no_solara.py` with 10 seeds in parallel via `ProcessPoolExecutor`. |
| `Run_no_solara.py` | Runs one parameter combo for N seeds and saves results to `results_saltelli/run_*.npz`. Each `.npz` contains the full H timeseries, utility series, Moran's I, neighbourhood income variance, and convergence steps across all seeds. |
| `Collect_outputs.py` | After all SLURM jobs finish, loads all `.npz` files and collapses each to a single scalar per output (mean over seeds, final timestep). Saves `Y_H.npy`, `Y_morans.npy`, `Y_nb_var.npy`, `Y_steps.npy`. Missing combos (failed jobs) are filled with the mean. |
| `Analyse_sobol.py` | Runs SALib's Sobol estimator on the Y vectors. Saves `sobol_indices.png` (bar chart of S1 and ST per output) and `sobol_indices.csv` (full results table with confidence intervals). |
| `SensitivityAnalysis.py`| Runs on the npz files generated through sampling to create H convergence plots. The files are saved in a results folder called `sobol_output` and the files created are called `sa_H_trajectories.png`, and `sa_H_trajectories_all.png`|
| `Heatmap.py`| Runs on `results_summary.csv` to create heatmaps, scatter plots, distributions of each convergence metrics (H-convergence, nb variance, and morans I). The files are saved in a results folder called `sobol_output` and the files created are called `sa_heatmap.png`, `sa_scatter.png`, `sa_heatmap_morans.png`, `sa_scatter_morans.png`, `sa_heatmap_nb_variance.png`,`sa_scatter_nb_variance.png`, `sa_morans_dist.png`, `sa_H_dist.png`, and `sa_nb_variance_dist.png`|
 
## Output files (not tracked by git)
 
All large output files are saved to `results_saltelli/` and ignored by git:
 
- `run_*.npz` — raw simulation results, one file per parameter combo (6144 files)
- `Y_*.npy` — collapsed scalar outputs for SALib
- `sobol_indices.png` — visualisation of S1 and ST indices
- `*.out` — SLURM log files

The following are tracked by git as they are needed to reproduce the analysis:
 
- `saltelli_params.json` — the exact parameter combinations used
- `saltelli_problem.json` — parameter bounds and names
- `sobol_indices.csv` — final Sobol indices with confidence intervals
