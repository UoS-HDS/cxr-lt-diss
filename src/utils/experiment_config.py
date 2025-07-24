"""
Centralized experiment configuration
Single point of truth for all experiment settings
"""

from pathlib import Path
from typing import Dict, Any


# EDIT THIS SECTION TO CONFIGURE YOUR EXPERIMENT
CURRENT_EXPERIMENT = {
    "task": "task1",  # task1, task2, or task3
    "model_type": "medvit",  # convnext, medvit, vit
    "model_name": "medvit",  # medvit, convnext_small.fb_in22k_ft_in1k_384
    "embedding": None,  # pubmedbert, umlsbert, None
    "zsl": 0,  # 0 for no ZSL, 1 for ZSL
    "loss_type": "asl",  # asl, ral
    "image_size": 1024,
    "lr": 5e-5,
    "batch_size": 8,
    "train_gpus": [4, 5],  # GPUs for training (backbone & fusion)
    "predict_gpu": 4,  # Single GPU for prediction
    "gpu_count": 2,  # Number of GPUs for training
    "mem": "48G",  # Memory for training
    "max_epochs": 150,
    "n_iterations": 3,  # Number of noisy student iterations
    "predict_type": "dev",  # dev or test, determines which dataset to use for prediction
    "project_dir": "/mnt/isilon1/na200/hds-diss",
    "apptainer_image": "pytorch_25.05-py3-uv",
    "docker_image": "pytorch_24.08-py3-uv",
}


def generate_experiment_name(config: Dict[str, Any]) -> str:
    """Generate experiment name from model_type, loss_type, and image_size"""
    model_short = config["model_type"]  # medvit or convnext
    loss = config["loss_type"]  # asl or ral
    size = config["image_size"]  # 1024, 384, etc.
    embedding = config["embedding"]  # pubmedbert, umlsbert, None
    lr = config["lr"]  # Learning rate

    if embedding:
        name = f"{model_short}+{loss}+{size}+{embedding}+{lr:.0e}"
    else:
        name = f"{model_short}+{loss}+{size}+{lr:.0e}"

    return name


def get_experiment_paths(config: Dict[str, Any], runtime: str) -> Dict[str, Path | str]:
    """Generate all paths for an experiment"""
    exp_name = generate_experiment_name(config)
    image_size = config["image_size"]
    task = config["task"]
    conf_matrix_name = "conf_matrix" if config["predict_type"] == "dev" else "conf_matrix_test"

    return {
        # Experiment name
        "exp_name": exp_name,
        # Checkpoint directories
        "checkpoint_dir": Path(
            f"checkpoints/backbones/{task}/ckpts/{image_size}/{exp_name}"
        ),
        "fusion_checkpoint_dir": Path(
            f"checkpoints/fusion/{task}/ckpts/{image_size}/{exp_name}"
        ),
        # Model files
        "model_path": Path(f"checkpoints/backbones/{task}/models/{exp_name}.pth"),
        "fusion_model_path": Path(f"checkpoints/fusion/{task}/models/{exp_name}.pth"),
        # Logging and outputs
        "tb_log_dir": Path(f".out/tb/{image_size}"),
        "submission_dir": Path(f"submissions/{task}/{exp_name}"),
        "conf_matrix_path": Path(f"submissions/{task}/{exp_name}/{conf_matrix_name}.svg"),
        # Pseudo-labels
        "pseudo_label_dir": Path(f".tmp/pseudolabels/{exp_name}"),
        "chexpert_pseudo_path": Path(
            f".tmp/pseudolabels/{exp_name}/chexpert/train_expanded_pseudo.csv"
        ),
        "vinbig_pseudo_path": Path(
            f".tmp/pseudolabels/{exp_name}/vinbig/train_expanded_pseudo.csv"
        ),
        "nih_pseudo_path": Path(
            f".tmp/pseudolabels/{exp_name}/nih/train_expanded_pseudo.csv"
        ),
        "pred_df_path": Path(config["pred_df_path"]),
        "config_dir": Path(f"configs/{task}"),
        # Backup directories
        "config_backup_dir": Path(f".tmp/configs/{exp_name}"),
        "scripts_backup_dir": Path(f".tmp/scripts/{runtime}/{exp_name}"),
        # Script directories
        "scripts_dir": Path(f"scripts/{runtime}/{task}"),
    }


def get_gpu_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Generate GPU configuration strings for Docker"""
    train_gpus = config["train_gpus"]
    train_gpu_str = ",".join(map(str, train_gpus))
    train_gpu_count = len(train_gpus)
    predict_gpu_str = str(train_gpus[0])

    return {
        "train_cuda_devices": train_gpu_str,
        "train_gpu_count": str(train_gpu_count),
        "predict_cuda_devices": predict_gpu_str,
        "predict_gpu_count": "1",
    }


def validate_experiment_config(config: Dict[str, Any]) -> None:
    """Validate experiment configuration"""
    required_keys = [
        "task",
        "model_type",
        "model_name",
        "loss_type",
        "image_size",
        "lr",
        "batch_size",
        "train_gpus",
        "predict_gpu",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if config["model_type"] not in ["convnext", "medvit"]:
        raise ValueError(f"Invalid model_type: {config['model_type']}")

    if config["loss_type"] not in ["asl", "ral"]:
        raise ValueError(f"Invalid loss_type: {config['loss_type']}")

    if config["task"] not in ["task1", "task2", "task3"]:
        raise ValueError(f"Invalid task: {config['task']}")

    if not isinstance(config["train_gpus"], list) or len(config["train_gpus"]) == 0:
        raise ValueError("train_gpus must be a non-empty list")

    if config["predict_gpu"] not in config["train_gpus"]:
        print(
            f"Warning: predict_gpu {config['predict_gpu']} not in train_gpus {config['train_gpus']}"
        )
