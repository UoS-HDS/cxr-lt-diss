#!/bin/bash
#SBATCH --job-name=backbone_teacher_predict
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=21

set -e

project_dir="/mnt/isilon1/$USER/hds-diss/"
ckpt_path="$1"

if [ -z "$ckpt_path" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

cd "$project_dir"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Execute training script
CUDA_VISIBLE_DEVICES=0 docker run --rm --gpus 1 --ipc=host \
    --user $(id -u):$(id -g) \
    -v "$project_dir":"$project_dir" \
    -w "$project_dir" \
    -e UV_PROJECT_ENVIRONMENT=/opt/venv/ \
    pytorch_24.08-py3-uv \
    bash -c "set -e; \
    cp -r configs/task1/* .tmp/configs/medvit+asl+1024/ && \
    STAGE=1 uv run --python 3.12.9 main.py predict --config .tmp/configs/medvit+asl+1024/config.yaml \
      --data.datamodule_cfg.predict_pseudo_label chexpert \
      --trainer.devices=1 --ckpt_path '$ckpt_path' && \
    STAGE=1 uv run --python 3.12.9 main.py predict --config .tmp/configs/medvit+asl+1024/config-nih.yaml \
      --trainer.devices=1 --ckpt_path '$ckpt_path' && \
    STAGE=1 uv run --python 3.12.9 main.py predict --config .tmp/configs/medvit+asl+1024/config-vinbig.yaml \
      --trainer.devices=1 --ckpt_path '$ckpt_path'" 2>&1 | tee logs/docker_${TIMESTAMP}.log

# Check if docker command succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Docker command failed!"
    exit 1
fi

# Confirmation message
echo "Job complete"