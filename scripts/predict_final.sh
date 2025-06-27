#!/bin/bash
#SBATCH --job-name=backbone_teacher_1
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mem=24G
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
    STAGE=2 uv run main.py predict --config configs/config-stage-2-pred.yaml > submissions/preds.log

# Confirmation message
echo "Job complete"