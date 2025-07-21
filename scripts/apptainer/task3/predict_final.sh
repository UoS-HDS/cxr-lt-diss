#!/bin/bash
#SBATCH --job-name=fusion_task1
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=21

set -e

project_dir="/sharedscratch/$USER/hds-diss"

cd "$project_dir"

# Remove existing venv
rm -rf .venv

# Execute training script
apptainer exec --nv \
    --home "$project_dir":$HOME \
    apptainer/pytorch_25.05-py3-uv.sif\
    uv sync -q && \
    STAGE=2 uv run main.py predict --config configs/config-stage-2-pred.yaml > submissions/preds.txt

# Confirmation message
echo "Job complete"