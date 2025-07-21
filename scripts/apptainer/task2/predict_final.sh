#!/bin/bash
#SBATCH --job-name=fusion_predict
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=21

set -e

project_dir="/sharedscratch/$USER/hds-diss"
ckpt_path="$1"
backbone_path="$2"

cd "$project_dir"

if [ -z "$ckpt_path" ]; then
    echo "Usage: $0 <checkpoint_path> <backbone_checkpoint_path>"
    exit 1
fi
if [ -z "$backbone_path" ]; then
    echo "Usage: $0 <checkpoint_path> <backbone_checkpoint_path>"
    exit 1
fi

cmd="STAGE=2 uv run main.py predict --config configs/task2/config-stage-2-pred.yaml --ckpt_path $ckpt_path --model.pretrained_path $backbone_path"

# Remove existing venv
rm -rf .venv

# Execute training script
apptainer exec --nv \
    --home "$project_dir":$HOME \
    apptainer/pytorch_25.05-py3-uv.sif\
    bash -c "uv sync -q && \
    $cmd > submissions/task2/convnext-384-default/results.txt"

# Confirmation message
echo "Job complete"