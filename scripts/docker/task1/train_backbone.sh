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
project_dir="//mnt/isilon1/$USER/hds-diss/"

cd "$project_dir"

cmd="STAGE=1 uv run --python 3.12.9 main.py fit --config .tmp/configs/medvit+asl+1024/config.yaml --data.datamodule_cfg.use_pseudo_label $use_pseudo"
if [ -n "$ckpt_path" ]; then
    cmd="$cmd --ckpt_path $ckpt_path"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Execute training script
CUDA_VISIBLE_DEVICES=4,5 docker run --rm --gpus 2 --ipc=host \
    --user $(id -u):$(id -g) \
    -v "$project_dir":"$project_dir" \
    -w "$project_dir" \
    -e UV_PROJECT_ENVIRONMENT=/opt/venv/ \
    pytorch_24.08-py3-uv \
    bash -c "set -e; cp -r configs/task1/* .tmp/configs/medvit+asl+1024/ && eval \"$cmd\"" 2>&1 | tee logs/docker_${TIMESTAMP}.log

# Check if docker command succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Docker command failed!"
    exit 1
fi

# Confirmation message
echo "Job complete"