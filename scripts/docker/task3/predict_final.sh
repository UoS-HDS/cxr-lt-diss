#!/bin/bash
#SBATCH --job-name=fusion_predict_task3
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=21

set -e

project_dir="/mnt/isilon1/$USER/hds-diss/"
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

cmd="STAGE=2 uv run --python 3.12.9 main.py predict --config /opt/configs/task3/config-stage-2-pred.yaml --ckpt_path $ckpt_path --model.pretrained_path $backbone_path"

# Execute training script
docker run --rm --gpus 1 --ipc=host \
    --user $(id -u):$(id -g) \
    -v "$project_dir":"$project_dir" \
    -w "$project_dir" \
    -e UV_PROJECT_ENVIRONMENT=.venv-docker \
    pytorch_24.08-py3-uv \
    bash -c "set -e; \
    cp -r configs/* /opt/configs/ && \
    $cmd > submissions/task3/results.txt"

# Check if docker command succeeded
if [ $? -ne 0 ]; then
    echo "Docker command failed!"
    exit 1
fi

# Confirmation message
echo "Job complete"