"""
Centralized experiment configuration
Single point of truth for all experiment settings
"""

from pathlib import Path
from typing import Dict, Any


# EDIT THIS SECTION TO CONFIGURE YOUR EXPERIMENT
CURRENT_EXPERIMENT = {
    "task": "task1",  # task1, task2, or task3
    "model_type": "convnext",  # convnext, medvit, vit
    "model_name": "convnext_small.fb_in22k_ft_in1k_384",  # medvit, convnext_small.fb_in22k_ft_in1k_384 (from timm)
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
    "iter": 0,  # Current NST iteration (0, 1, 2...n-1, auto-passed during runs)
    "predict_type": "dev",  # dev or test, determines which dataset to use for prediction
    "project_dir": "/mnt/isilon1/na200/hds-diss",
    "apptainer_image": "pytorch_25.05-py3-uv",
    "docker_image": "pytorch_24.08-py3-uv",
}


def generate_experiment_name(config: Dict[str, Any]) -> str:
    """Generate experiment name from model_type, loss_type, and image_size"""
    model_short = config["model_type"]
    loss = config["loss_type"]
    size = config["image_size"]
    embedding = config["embedding"]
    lr = config["lr"]

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
    conf_matrix_name = (
        "conf_matrix" if config["predict_type"] == "dev" else "conf_matrix_test"
    )

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
        "conf_matrix_path": Path(
            f"submissions/{task}/{exp_name}/{conf_matrix_name}.svg"
        ),
        # Pseudo-labels
        "pseudo_label_dir": Path(f".tmp/pseudolabels/{task}/{exp_name}"),
        "chexpert_pseudo_path": Path(
            f".tmp/pseudolabels/chexpert/{task}/{exp_name}/train_expanded_pseudo.csv"
        ),
        "vinbig_pseudo_path": Path(
            f".tmp/pseudolabels/vinbig/{task}/{exp_name}/train_expanded_pseudo.csv"
        ),
        "nih_pseudo_path": Path(
            f".tmp/pseudolabels/nih/{task}/{exp_name}/train_expanded_pseudo.csv"
        ),
        "pred_df_path": Path(config["pred_df_path"]),
        "configs_dir": Path(f".tmp/configs/{task}/{exp_name}"),
        "scripts_dir": Path(f".tmp/scripts/{runtime}/{task}/{exp_name}"),
    }


def create_experiment_directories(paths: Dict[str, Path]) -> None:
    """Create all necessary directories for an experiment"""

    directories_to_create = [
        paths["checkpoint_dir"],
        paths["fusion_checkpoint_dir"],
        paths["submission_dir"],
        paths["pseudo_label_dir"] / "chexpert",
        paths["pseudo_label_dir"] / "vinbig",
        paths["pseudo_label_dir"] / "nih",
        paths["configs_dir"],
        paths["scripts_dir"],
        paths["tb_log_dir"],
        Path("logs"),
    ]

    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)

    print("✅ Created experiment directories")


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
