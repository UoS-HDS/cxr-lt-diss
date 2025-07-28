"""
Docker script generator for experiments
Dynamically generates all Docker scripts based on experiment settings
"""

from pathlib import Path
from typing import Dict, Any
from .experiment_config import get_gpu_config


def generate_train_backbone_script(
    config: Dict[str, Any], paths: Dict[str, Path]
) -> str:
    """Generate train_backbone.sh script"""
    gpu_config = get_gpu_config(config)
    docker_image = config["docker_image"]

    return f"""#!/bin/bash
set -e

use_pseudo="$1"
ckpt_path="$2"
project_dir="//mnt/isilon1/$USER/hds-diss/"

cd "$project_dir"

cmd="STAGE=1 uv run --python 3.12.9 main.py fit --config {paths["config_backup_dir"]}/config.yaml --data.datamodule_cfg.use_pseudo_label $use_pseudo"
if [ -n "$ckpt_path" ]; then
    cmd="$cmd --ckpt_path $ckpt_path"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Execute training script
CUDA_VISIBLE_DEVICES={gpu_config["train_cuda_devices"]} docker run --rm --gpus '"device={gpu_config["train_cuda_devices"]}"' --ipc=host \\
    --user $(id -u):$(id -g) \\
    -v "$project_dir":"$project_dir" \\
    -w "$project_dir" \\
    -e UV_PROJECT_ENVIRONMENT=/opt/venv/ \\
    -e NCCL_DEBUG=INFO \\
    -e NCCL_IB_DISABLE=1 \\
    -e NCCL_P2P_DISABLE=1 \\
    {docker_image} \\
    bash -c "set -e; eval \\"$cmd\\"" 2>&1 | tee logs/docker_${{TIMESTAMP}}.log

# Check if docker command succeeded
if [ ${{PIPESTATUS[0]}} -ne 0 ]; then
    echo "Docker command failed!"
    exit 1
fi

# Confirmation message
echo "Job complete"
"""


def generate_predict_pseudo_labels_script(
    config: Dict[str, Any], paths: Dict[str, Path]
) -> str:
    """Generate predict_pseudo_labels.sh script"""
    gpu_config = get_gpu_config(config)
    docker_image = config["docker_image"]

    return f"""#!/bin/bash

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
CUDA_VISIBLE_DEVICES={gpu_config["predict_cuda_devices"]} docker run --rm --gpus '"device={gpu_config["predict_cuda_devices"]}"' --ipc=host \\
    --user $(id -u):$(id -g) \\
    -v "$project_dir":"$project_dir" \\
    -w "$project_dir" \\
    -e UV_PROJECT_ENVIRONMENT=/opt/venv/ \\
    {docker_image} \\
    bash -c "set -e; \\
    STAGE=1 uv run --python 3.12.9 main.py predict --config {paths["config_backup_dir"]}/config.yaml \\
      --data.datamodule_cfg.predict_pseudo_label chexpert \\
      --trainer.devices=1 --ckpt_path '$ckpt_path' && \\
    STAGE=1 uv run --python 3.12.9 main.py predict --config {paths["config_backup_dir"]}/config-nih.yaml \\
      --trainer.devices=1 --ckpt_path '$ckpt_path' --data.datamodule_cfg.predict_pseudo_label nih && \\
    STAGE=1 uv run --python 3.12.9 main.py predict --config {paths["config_backup_dir"]}/config-vinbig.yaml \\
      --trainer.devices=1 --ckpt_path '$ckpt_path' --data.datamodule_cfg.predict_pseudo_label vinbig" 2>&1 | tee logs/docker_${{TIMESTAMP}}.log

# Check if docker command succeeded
if [ ${{PIPESTATUS[0]}} -ne 0 ]; then
    echo "Docker command failed!"
    exit 1
fi

# Confirmation message
echo "Job complete"
"""


def generate_train_fusion_script(config: Dict[str, Any], paths: Dict[str, Path]) -> str:
    """Generate train_fusion.sh script"""
    gpu_config = get_gpu_config(config)
    docker_image = config["docker_image"]

    return f"""#!/bin/bash

set -e

cd /mnt/isilon1/$USER/hds-diss/

backbone_path="$1"
if [ -z "$backbone_path" ]; then
    echo "Usage: $0 <backbone_checkpoint_path>"
    exit 1
fi

cmd="STAGE=2 uv run --python 3.12.9 main.py fit --config {paths["config_backup_dir"]}/config-stage-2.yaml --model.pretrained_path $backbone_path"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Execute training script
CUDA_VISIBLE_DEVICES={gpu_config["train_cuda_devices"]} docker run --rm --gpus '"device={gpu_config["train_cuda_devices"]}"' --ipc=host \\
    --user $(id -u):$(id -g) \\
    -v "/mnt/isilon1/$USER/hds-diss/":"/mnt/isilon1/$USER/hds-diss/" \\
    -w "/mnt/isilon1/$USER/hds-diss/" \\
    -e UV_PROJECT_ENVIRONMENT=/opt/venv/ \\
    -e NCCL_DEBUG=INFO \\
    -e NCCL_IB_DISABLE=1 \\
    -e NCCL_P2P_DISABLE=1 \\
    {docker_image} \\
    bash -c "set -e; $cmd" 2>&1 | tee logs/docker_${{TIMESTAMP}}.log

# Check if docker command succeeded
if [ ${{PIPESTATUS[0]}} -ne 0 ]; then
    echo "Docker command failed!"
    exit 1
fi

# Confirmation message
echo "Job complete"
"""


def generate_predict_final_script(
    config: Dict[str, Any], paths: Dict[str, Path]
) -> str:
    """Generate predict_final.sh script"""
    gpu_config = get_gpu_config(config)
    docker_image = config["docker_image"]
    if config["predict_type"] == "dev":
        res_file = "results.txt"
    else:
        res_file = "results_test.txt"

    return f"""#!/bin/bash

set -e

project_dir="/mnt/isilon1/$USER/hds-diss/"
ckpt_path="$1"

if [ -z "$ckpt_path" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

cmd="STAGE=2 uv run --python 3.12.9 main.py predict --config {paths["config_backup_dir"]}/config-stage-2-pred.yaml --ckpt_path $ckpt_path"

# Execute training script
CUDA_VISIBLE_DEVICES={gpu_config["predict_cuda_devices"]} docker run --rm --gpus '"device={gpu_config["predict_cuda_devices"]}"' --ipc=host \\
    --user $(id -u):$(id -g) \\
    -v "$project_dir":"$project_dir" \\
    -w "$project_dir" \\
    -e UV_PROJECT_ENVIRONMENT=/opt/venv/ \\
    {docker_image} \\
    bash -c "set -e; \\
    $cmd > {paths["submission_dir"]}/{res_file}"

# Check if docker command succeeded
if [ $? -ne 0 ]; then
    echo "Docker command failed!"
    exit 1
fi

# Confirmation message
echo "Job complete"
"""


def write_script_files(config: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """Generate and write all Docker script files"""

    # Create scripts directory
    scripts_dir = paths["scripts_backup_dir"]
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Generate all scripts
    scripts = {
        "train_backbone.sh": generate_train_backbone_script(config, paths),
        "predict_pseudo_labels.sh": generate_predict_pseudo_labels_script(
            config, paths
        ),
        "train_fusion.sh": generate_train_fusion_script(config, paths),
        "predict_final.sh": generate_predict_final_script(config, paths),
    }

    # Write all script files
    for filename, script_content in scripts.items():
        script_path = scripts_dir / filename
        with open(script_path, "w") as f:
            f.write(script_content)

        # Make executable
        script_path.chmod(0o755)

        print(f"✅ Generated: {script_path}")
