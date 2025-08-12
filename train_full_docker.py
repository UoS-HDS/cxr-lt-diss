#!/usr/bin/env python3
"""
Centralized Experiment Runner
Single point of control for all experiments

Based on the original train_full_docker.py but with centralized configuration.
Edit src/utils/experiment_config.py to change experiment settings.
"""

import subprocess
import signal
import sys
import os
from pathlib import Path
import argparse
from typing import Dict, Any

from src.utils.runner_utils import run_full_pipeline, run_predict_only, setup_experiment
from src.utils.experiment_config import CURRENT_EXPERIMENT, get_experiment_paths


# Global variable to track running processes
running_processes = []


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully by terminating all running Docker containers."""
    print("\\nInterrupt received. Terminating all running Docker containers...")
    for process in running_processes:
        if process.poll() is None:  # Process is still running
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def run_docker_script(
    script_path: Path,
    ckpt_path: Path | None = None,
    use_pseudo: int = 0,
    predict: bool = False,
) -> subprocess.Popen:
    """
    Run a Docker script directly without SLURM.
    """
    # Make script executable
    os.chmod(script_path, 0o755)

    if predict:
        if ckpt_path:
            cmd = [str(script_path), str(ckpt_path)]
        else:
            cmd = [str(script_path), "null"]
    else:
        if ckpt_path and use_pseudo:
            cmd = [str(script_path), str(use_pseudo), str(ckpt_path)]
        elif ckpt_path:
            cmd = [str(script_path), str(ckpt_path)]
        else:
            cmd = [str(script_path), str(use_pseudo)]

    print(f"Running command: {' '.join(cmd)}")

    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Add to global list for signal handling
    running_processes.append(process)

    return process


def wait_for_process_completion(process: subprocess.Popen, job_name: str):
    """
    Wait for a process to complete and stream its output.
    """
    print(f"Starting {job_name}...")

    # Stream output in real-time
    for line in iter(process.stdout.readline, ""):  # type: ignore
        print(line.rstrip())

    # Wait for process to complete
    return_code = process.wait()

    # Remove from global list
    if process in running_processes:
        running_processes.remove(process)

    if return_code != 0:
        raise RuntimeError(f"{job_name} failed with return code {return_code}")

    print(f"{job_name} completed successfully.")


def run_job(
    script_path: Path,
    job_name: str,
    ckpt_path: Path | None = None,
    use_pseudo: int = 0,
    predict: bool = False,
):
    """
    Run a job and wait for completion.
    """
    process = run_docker_script(
        script_path=script_path,
        ckpt_path=ckpt_path,
        use_pseudo=use_pseudo,
        predict=predict,
    )
    wait_for_process_completion(process, job_name)


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

    if config["model_type"] not in ["convnext", "medvit", "vit", "maxvit"]:
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Centralized Experiment Runner")
    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="Whether to run in prediction mode only",
    )
    parser.add_argument(
        "--fusion_only",
        action="store_true",
        help="Whether to run stage 2 fusion training only",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to run (task1, task2, or task3)",
        choices=["task1", "task2", "task3"],
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Model type",
        choices=["convnext", "medvit", "vit", "maxvit"],
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name (e.g., convnext_small.fb_in22k_ft_in1k_384)",
        required=True,
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="Embedding type (pubmedbert, umlsbert)",
        choices=["pubmedbert", "umlsbert"],
        default=None,
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        help="Loss function (asl or ral)",
        choices=["asl", "ral"],
        required=True,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        help="Image size for training",
        required=True,
    )
    parser.add_argument("--lr", type=float, help="Learning rate", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size", required=True)
    parser.add_argument(
        "--train_gpus",
        type=int,
        nargs="+",
        help="List of GPUs for training",
        required=True,
    )
    parser.add_argument("--predict_gpu", type=int, help="GPU for prediction")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs")
    parser.add_argument(
        "--n_iterations",
        type=int,
        help="Number of noisy student iterations",
        default=3,
    )
    parser.add_argument(
        "--predict_type",
        type=str,
        help="Prediction type (dev or test)",
        default="dev",
        choices=["dev", "test"],
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        help="Project root directory",
        required=True,
    )
    parser.add_argument(
        "--docker_image",
        type=str,
        help="Docker image name",
        required=True,
    )

    args = parser.parse_args()

    # Get experiment configuration
    config = CURRENT_EXPERIMENT.copy()
    if args.task == "task3":
        config["zsl"] = 1

    # Override config with command-line arguments if provided
    for key, value in vars(args).items():
        if value is not None and key != "predict_only":
            config[key] = value

    predict_type = config.get("predict_type", "dev")

    PRED_DF_DIR = Path("data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/")
    if predict_type == "dev":
        pred_df_path = PRED_DF_DIR / f"development_labeled_{config['task']}.csv"
    else:
        pred_df_path = PRED_DF_DIR / f"test_labeled_{config['task']}.csv"

    config["pred_df_path"] = str(pred_df_path)
    paths = get_experiment_paths(config, "docker")

    print("=" * 60)
    print("🎯 CENTRALIZED EXPERIMENT RUNNER")
    print("=" * 60)

    # Setup experiment
    setup_experiment(config, paths, "docker", validate_experiment_config)  # type: ignore

    if args.predict_only:
        run_predict_only(config, paths, run_job)  # type: ignore
        return

    # Run the full pipeline
    print("\n🏃 Starting training pipeline...")
    run_full_pipeline(
        config=config,
        paths=paths,  # type: ignore
        run_job=run_job,
        valid_func=validate_experiment_config,
        runtime="docker",
        fusion_only=args.fusion_only,
    )

    print("\n🎉 Experiment completed successfully!")
    print(f"📁 Results saved to: {paths['submission_dir']}")
    print(f"🎯 Confusion matrix: {paths['conf_matrix_path']}")


if __name__ == "__main__":
    main()
