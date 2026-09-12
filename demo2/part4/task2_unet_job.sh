#!/bin/bash
#SBATCH --job-name=oasis-unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=unet_%j.out
#SBATCH --error=unet_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate torch

# Submit from part4; pass --mode inference to load saved weights.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
python -u task2_unet.py "$@"
