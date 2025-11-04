#!/usr/bin/env bash

#SBATCH --job-name=hyperparam_heat_eq.py
#SBATCH --output=logs/hyperparam_heat_eq.out
#SBATCH --error=logs/hyperparam_heat_eq.err

#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1  

#SBATCH --mem=8G      
#SBATCH --time=0:60:00
#SBATCH --output=%N-%j.out

module load python/3.10
module load cuda

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index numpy

pip install --no-index pandas

sbatch run_job.sbatch