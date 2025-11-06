#!/bin/bash
#SBATCH --account=gerolinlab
#SBATCH --job-name=kl_hyperparameter
#SBATCH --array=0-440
#SBATCH --output=gradient_flow/kl/logs/slurm_%A_%a.out
#SBATCH --error=gradient_flow/kl/logs/slurm_%A_%a.err
#SBATCH --partition=General
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=16G

# Limit GPU usage per job (e.g., use only GPU 0, and restrict memory to 10%)
# export CUDA_VISIBLE_DEVICES=2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mairiconda
python kl_div.py