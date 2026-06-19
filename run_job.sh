#!/bin/bash
#SBATCH --job-name=schelling
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-9                              # 10 tasks (seeds 0–9), runs in parallel
#SBATCH --output=output/schelling_%A_%a.out      # %A = array job ID, %a = task index

mkdir -p output

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /gpfs/home4/scur0391/projects/ABM/.venv/bin/activate

# Pass the task index as the seed
uv run python /gpfs/home4/scur0391/projects/ABM/Run_no_solara.py --seed $SLURM_ARRAY_TASK_ID
