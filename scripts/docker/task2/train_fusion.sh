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

cd /mnt/isilon1/$USER/hds-diss/

backbone_path="$1"
if [ -z "$backbone_path" ]; then
    echo "Usage: $0 <backbone_checkpoint_path>"
    exit 1
fi

cmd="STAGE=2 uv run --python 3.12.9 main.py fit --config .tmp/configs/convnext+asl+384/config-stage-2.yaml --model.pretrained_path $backbone_path"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Execute training script
CUDA_VISIBLE_DEVICES=0,1 docker run --rm --gpus 2 --ipc=host \
    --user $(id -u):$(id -g) \
    -v "/mnt/isilon1/$USER/hds-diss/":"/mnt/isilon1/$USER/hds-diss/" \
    -w "/mnt/isilon1/$USER/hds-diss/" \
    -e UV_PROJECT_ENVIRONMENT=/opt/venv/ \
    pytorch_24.08-py3-uv \
    bash -c "set -e; cp -r configs/task1/* .tmp/configs/convnext+asl+384/ && $cmd" 2>&1 | tee logs/docker_${TIMESTAMP}.log

# Check if docker command succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Docker command failed!"
    exit 1
fi

# Confirmation message
echo "Job complete"
