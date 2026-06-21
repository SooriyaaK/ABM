#!/bin/bash
#SBATCH --job-name=schelling_test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-4                               # just 5 combos
#SBATCH --output=output/schelling_test_%A_%a.out

mkdir -p output
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /gpfs/home4/scur0391/projects/ABM/.venv/bin/activate
uv run python /gpfs/home4/scur0391/projects/ABM/Run_no_solara.py \
--combo-idx $SLURM_ARRAY_TASK_ID \
--n-seeds 10 \
--max-steps 50 \
--params-file /gpfs/home4/scur0391/projects/ABM/params_test.json