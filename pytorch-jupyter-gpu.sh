#!/bin/bash
#SBATCH --job-name=jupyter_gpu
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=21

# Navigate to project directory
cd /sharedscratch/na200/hds-diss

# Execute training script
apptainer exec --nv --home /sharedscratch/na200/hds-diss:$HOME ./apptainer/pytorch_25.05-py3-uv.sif \
    jupyter lab --notebook-dir=$HOME --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.allow_remote_access=True

# Confirmation message
echo "Job complete"