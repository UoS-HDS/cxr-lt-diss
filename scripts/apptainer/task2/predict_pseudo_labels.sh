#!/bin/bash
#SBATCH --job-name=backbone_teacher_predict_task2
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=21

set -e

project_dir="/sharedscratch/$USER/hds-diss"
ckpt_path="$1"

if [ -z "$ckpt_path" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

cd "$project_dir"

# Remove existing venv
rm -rf .venv

# Execute training script
apptainer exec --nv \
    --home "$project_dir":$HOME \
    apptainer/pytorch_25.05-py3-uv.sif\
    bash -c "uv sync -q && \
    STAGE=1 uv run main.py predict --config configs/task2/config.yaml \
      --data.datamodule_cfg.predict_pseudo_label chexpert \
      --trainer.devices=1 --ckpt_path '$ckpt_path' && \
    STAGE=1 uv run main.py predict --config configs/task2/config-nih.yaml \
      --trainer.devices=1 --ckpt_path '$ckpt_path' && \
    STAGE=1 uv run main.py predict --config configs/task2/config-vinbig.yaml \
      --trainer.devices=1 --ckpt_path '$ckpt_path'"

# Confirmation message
echo "Job complete"