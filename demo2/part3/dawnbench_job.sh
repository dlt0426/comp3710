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

# Submit from the part3 directory. Forward arguments, e.g. --mode demo.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
python -u dawnbench.py "$@"
