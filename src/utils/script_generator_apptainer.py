from pathlib import Path
from typing import Any, Dict


def _get_slurm_header(
    job_name: str,
    gpu_count: int,
    mem: str = "48G",
    time: str = "72:00:00",
    partition: str = "gpu",
) -> str:
    """Generates a standard SLURM header for job scripts."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm_{job_name}_%j.out
#SBATCH --error=logs/slurm_{job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu_count}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=21
"""


def _generate_train_backbone_script(
    config: Dict[str, Any],
    paths: Dict[str, Path],
) -> str:
    """Generates the script for training the backbone model with Apptainer."""
    task = config["task"]
    job_name = f"train_backbone_{task}"
    gpu_count = int(config["gpu_count"])
    slurm_header = _get_slurm_header(job_name, gpu_count)
    project_dir = config["project_dir"]
    apptainer_image = config["apptainer_image"]
    config_path = paths["config_backup_dir"] / "config.yaml"

    return f"""{slurm_header}
set -e

use_pseudo="$1"
ckpt_path="$2"
project_dir="{project_dir}"

cd "$project_dir"

cmd="STAGE=1 uv run --python 3.12.9 main.py fit --config {config_path} --data.datamodule_cfg.use_pseudo_label $use_pseudo"
if [ -n "$ckpt_path" ]; then
    cmd="$cmd --ckpt_path '$ckpt_path'"
fi

echo "Executing command: $cmd"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

apptainer exec --nv \\
    --bind "$project_dir":"$project_dir" \\
    {apptainer_image} \\
    bash -c "uv sync -q && eval \\"$cmd\\""

if [ ${{PIPESTATUS[0]}} -ne 0 ]; then
    echo "Apptainer command failed!"
    exit 1
fi

echo "Job complete"
"""


def _generate_predict_pseudo_script(
    config: Dict[str, Any],
    paths: Dict[str, Path],
) -> str:
    """Generates the script for predicting pseudo-labels with Apptainer."""
    task = config["task"]
    job_name = f"pseudo_predict_{task}"
    gpu_count = int(config["gpu_count"])
    slurm_header = _get_slurm_header(job_name, gpu_count, mem="24G")
    project_dir = config["project_dir"]
    apptainer_image = config["apptainer_image"]
    config_path = paths["config_backup_dir"] / "config.yaml"
    nih_config_path = paths["config_backup_dir"] / "config-nih.yaml"
    vinbig_config_path = paths["config_backup_dir"] / "config-vinbig.yaml"

    return f"""{slurm_header}
set -e

ckpt_path="$1"
project_dir="{project_dir}"

if [ -z "$ckpt_path" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

cd "$project_dir"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

apptainer exec --nv \\
    --bind "$project_dir":"$project_dir" \\
    {apptainer_image} \\
    bash -c "set -e; \\
    uv sync -q && \\
    STAGE=1 uv run --python 3.12.9 main.py predict --config {config_path} --data.datamodule_cfg.predict_pseudo_label chexpert --trainer.devices=1 --ckpt_path '$ckpt_path' && \\
    STAGE=1 uv run --python 3.12.9 main.py predict --config {nih_config_path} --data.datamodule_cfg.predict_pseudo_label nih --trainer.devices=1 --ckpt_path '$ckpt_path' && \\
    STAGE=1 uv run --python 3.12.9 main.py predict --config {vinbig_config_path} --data.datamodule_cfg.predict_pseudo_label vinbig --trainer.devices=1 --ckpt_path '$ckpt_path'"

if [ ${{PIPESTATUS[0]}} -ne 0 ]; then
    echo "Apptainer command failed!"
    exit 1
fi

echo "Job complete"
"""


def _generate_train_fusion_script(
    config: Dict[str, Any],
    paths: Dict[str, Path],
) -> str:
    """Generates the script for training the fusion model with Apptainer."""
    task = config["task"]
    job_name = f"train_fusion_{task}"
    gpu_count = int(config["gpu_count"])
    slurm_header = _get_slurm_header(job_name, gpu_count)
    project_dir = config["project_dir"]
    apptainer_image = config["apptainer_image"]
    config_path = paths["config_backup_dir"] / "config-stage-2.yaml"

    return f"""{slurm_header}
set -e

backbone_path="$1"
project_dir="{project_dir}"

if [ -z "$backbone_path" ]; then
    echo "Usage: $0 <backbone_checkpoint_path>"
    exit 1
fi

cd "$project_dir"
cmd="STAGE=2 uv run --python 3.12.9 main.py fit --config {config_path} --model.pretrained_path '$backbone_path'"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

apptainer exec --nv \\
    --bind "$project_dir":"$project_dir" \\
    {apptainer_image} \\
    bash -c "uv sync -q && $cmd"

if [ ${{PIPESTATUS[0]}} -ne 0 ]; then
    echo "Apptainer command failed!"
    exit 1
fi

echo "Job complete"
"""


def _generate_predict_final_script(
    config: Dict[str, Any],
    paths: Dict[str, Path],
) -> str:
    """Generates the script for final prediction with Apptainer."""
    task = config["task"]
    job_name = f"final_predict_{task}"
    gpu_count = int(config["gpu_count"])
    slurm_header = _get_slurm_header(job_name, gpu_count)
    project_dir = config["project_dir"]
    apptainer_image = config["apptainer_image"]
    config_path = paths["config_backup_dir"] / "config-stage-2-pred.yaml"
    submission_path = paths["submission_dir"] / "results.txt"

    return f"""{slurm_header}
set -e

ckpt_path="$1"
backbone_path="$2"
project_dir="{project_dir}"

if [ -z "$ckpt_path" ] || [ -z "$backbone_path" ]; then
    echo "Usage: $0 <checkpoint_path> <backbone_checkpoint_path>"
    exit 1
fi

cd "$project_dir"
cmd="STAGE=2 uv run --python 3.12.9 main.py predict --config {config_path} --ckpt_path '$ckpt_path' --model.pretrained_path '$backbone_path'"

apptainer exec --nv \\
    --bind "$project_dir":"$project_dir" \\
    {apptainer_image} \\
    bash -c "uv sync -q && $cmd > {submission_path}"

if [ $? -ne 0 ]; then
    echo "Apptainer command failed!"
    exit 1
fi

echo "Final predictions saved to {submission_path}"
echo "Job complete"
"""


def write_apptainer_script_files(
    config: Dict[str, Any],
    paths: Dict[str, Path],
) -> None:
    """Writes all necessary Apptainer bash scripts for an experiment."""
    script_dir = paths["scripts_backup_dir"]
    script_dir.mkdir(parents=True, exist_ok=True)

    scripts_to_generate = {
        "train_backbone.sh": _generate_train_backbone_script(config, paths),
        "predict_pseudo_labels.sh": _generate_predict_pseudo_script(config, paths),
        "train_fusion.sh": _generate_train_fusion_script(config, paths),
        "predict_final.sh": _generate_predict_final_script(config, paths),
    }

    for filename, generator_func in scripts_to_generate.items():
        script_content = generator_func(config, paths)
        with open(script_dir / filename, "w") as f:
            f.write(script_content)
        # Make script executable
        (script_dir / filename).chmod(0o755)

    print(f"✅ Apptainer scripts generated in {script_dir}")
