#!/bin/bash
#SBATCH --job-name=fusion_train
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=21

set -e

cd /sharedscratch/$USER/hds-diss/

# Remove existing venv
rm -rf .venv

backbone_path="$1"
if [ -z "$backbone_path" ]; then
    echo "Usage: $0 <backbone_checkpoint_path>"
    exit 1
fi

cmd="STAGE=2 uv run main.py fit --config configs/task1/config-stage-2.yaml --model.pretrained_path $backbone_path"

# Execute training script
apptainer exec --nv \
    --home /sharedscratch/$USER/hds-diss/:$HOME \
    apptainer/pytorch_25.05-py3-uv.sif\
    bash -c "uv sync -q && $cmd"

# Confirmation message
echo "Job complete"