#!/bin/bash
#SBATCH --job-name=backbone_teacher
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=21

set -e

use_pseudo="$1"
ckpt_path="$2"
project_dir="/sharedscratch/$USER/hds-diss"

cd "$project_dir"

# Remove existing venv
rm -rf .venv

cmd="STAGE=1 uv run main.py fit --config configs/task1/config.yaml --data.datamodule_cfg.use_pseudo_label $use_pseudo"
if [ -n "$ckpt_path" ]; then
    cmd="$cmd --ckpt_path $ckpt_path"
fi

# Execute training script
apptainer exec --nv \
    --home "$project_dir":$HOME \
    apptainer/pytorch_25.05-py3-uv.sif\
    bash -c "uv sync -q && eval $cmd"

# Confirmation message
echo "Job complete"