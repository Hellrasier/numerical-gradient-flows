#!/bin/bash
#SBATCH --account=gerolinlab
#SBATCH --job-name=jko_hpo_optuna
#SBATCH --output=logs/jko_hpo_%j.out
#SBATCH --error=logs/jko_hpo_%j.err
#SBATCH --partition=General
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# GPU memory management
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jko_optuna

# Create necessary directories
mkdir -p logs
mkdir -p results

# Run the hyperparameter optimization
python jko_hpo_optuna.py

echo "Job completed at $(date)"