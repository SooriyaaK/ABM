#!/bin/bash
#SBATCH --job-name=schelling
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=output/schelling_%j.out   # %j = job ID

# Ensure output directory exists
mkdir -p output

# Load your environment
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate your venv
source /gpfs/home4/scur0391/projects/ABM/.venv/bin/activate

# Run
uv run python /gpfs/home4/scur0391/projects/ABM/Run_no_solara.py
