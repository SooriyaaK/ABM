#!/bin/bash
#SBATCH --job-name=schelling_test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4G
#SBATCH --array=0-6143%200                            # 5 test combos (params_test.json)
#SBATCH --output=results_saltelli/schelling_%A_%a.out  # %A = array job ID, %a = task index

mkdir -p results_saltelli  # instead of mkdir -p output

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /gpfs/home4/scur0391/projects/ABM/.venv/bin/activate

# Fewer seeds + steps than the full run, so the test finishes fast.
uv run python /gpfs/home4/scur0391/projects/ABM/Run_no_solara.py \
    --combo-idx $SLURM_ARRAY_TASK_ID \
    --n-seeds 10 \
    --max-steps 500 \
    --params-file /gpfs/home4/scur0391/projects/ABM/results_saltelli/saltelli_params.json
