#!/bin/bash
#SBATCH --job-name=dawnbench
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=dawnbench_%j.out
#SBATCH --error=dawnbench_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate torch

python part3_dawnbench.py
