#!/bin/bash
#SBATCH --job-name=backbone_teacher_1
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --time=60:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=21

set -e

cd /sharedscratch/$USER/hds-diss/

# Remove existing venv
rm -rf .venv

# Load Apptainer module
# module load apptainer

# Execute training script
apptainer exec --nv \
    --home /sharedscratch/$USER/hds-diss/:$HOME \
    apptainer/pytorch_25.05-py3-uv.sif\
    uv sync -q && \
    STAGE=2 uv run main.py fit --config configs/config-stage-2.yaml

# Confirmation message
echo "Job complete"