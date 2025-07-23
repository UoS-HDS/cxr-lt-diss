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
from typing import Dict, Any
import argparse

# Import our utilities
from src.utils.experiment_config import (
    CURRENT_EXPERIMENT,
    get_experiment_paths,
    validate_experiment_config,
)
from src.utils.config_generator import write_config_files
from src.utils.script_generator import write_script_files
from src.utils.directory_manager import create_experiment_directories


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


def get_best_ckpt(ckpt_dir: Path) -> Path | None:
    """
    Get the most recent checkpoint file from the specified directory.
    """

    def best_ap_idx(ckpts: list[Path]) -> int:
        max_ap, max_idx = 0.0, 0
        for i, ckpt in enumerate(ckpts):
            if ckpt.stem.find("last") != -1:
                continue
            val_ap = ckpt.stem.split("=")[-1]
            if float(val_ap) >= max_ap:
                max_ap = float(val_ap)
                max_idx = i

        return max_idx

    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}.")

    best_ckpt = ckpt_files[best_ap_idx(ckpt_files)]

    return best_ckpt


def run_docker_script(
    script_path: Path,
    ckpt_path: Path | None = None,
    use_pseudo: int = 0,
    predict: bool = False,
    extras: list[str | Path] = [],
) -> subprocess.Popen:
    """
    Run a Docker script directly without SLURM.
    """
    # Make script executable
    os.chmod(script_path, 0o755)

    if not ckpt_path:
        cmd = [str(script_path), str(use_pseudo)]
    else:
        if predict and extras:
            cmd = [str(script_path), str(ckpt_path), *[str(x) for x in extras]]
        elif predict:
            cmd = [str(script_path), str(ckpt_path)]
        elif not use_pseudo:
            cmd = [str(script_path), str(ckpt_path)]
        else:
            cmd = [str(script_path), str(use_pseudo), str(ckpt_path)]

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
    extras: list[str | Path] = [],
):
    """
    Run a job and wait for completion.
    """
    process = run_docker_script(
        script_path=script_path,
        ckpt_path=ckpt_path,
        use_pseudo=use_pseudo,
        predict=predict,
        extras=extras,
    )
    wait_for_process_completion(process, job_name)


def setup_experiment(config: Dict[str, Any], paths: Dict[str, Path]):
    """Set up all experiment files and directories"""

    exp_name = paths["exp_name"]
    print(f"🚀 Setting up experiment: {exp_name}")
    print(
        f"📊 Model: {config['model_type']} | Loss: {config['loss_type']} | Size: {config['image_size']} | GPUs: {config['train_gpus']}"
    )

    # 1. Validate configuration
    validate_experiment_config(config)

    # 2. Create directories
    create_experiment_directories(paths)

    # 3. Generate configuration files
    print("\n📝 Generating configuration files...")
    write_config_files(config, paths)

    # 4. Generate script files
    print("\n🔧 Generating Docker scripts...")
    write_script_files(config, paths)

    print("\n✅ Experiment setup complete!")


def run_full_pipeline(config: Dict[str, Any], paths: Dict[str, Path]):
    """Run the complete training pipeline"""

    SCRIPTS_DIR = paths["scripts_backup_dir"]

    # Use locked scripts instead of original ones
    train_script = SCRIPTS_DIR / "train_backbone.sh"
    predict_script = SCRIPTS_DIR / "predict_pseudo_labels.sh"
    fusion_script = SCRIPTS_DIR / "train_fusion.sh"
    predict_fusion_script = SCRIPTS_DIR / "predict_final.sh"

    scripts_found = all(
        script.exists()
        for script in [
            train_script,
            predict_script,
            fusion_script,
            predict_fusion_script,
        ]
    )
    if not scripts_found:
        raise FileNotFoundError(f"Required scripts not found in {SCRIPTS_DIR}.")

    print(f"Using scripts from: {SCRIPTS_DIR}")

    N_ITER = config.get("n_iterations", 3)

    try:
        for idx in range(N_ITER + 1):
            if idx == 0:
                print("INITIAL TRAINING RUN WITH EXISTING LABELS...")
                run_job(train_script, "Initial Training")
                continue

            print(f"\nNOISY STUDENT ITERATION {idx}")
            ckpt = get_best_ckpt(paths["checkpoint_dir"])

            print(f"USING CHECKPOINT: {ckpt}")
            run_job(
                predict_script,
                f"Predicting pseudolabels (iter {idx})",
                ckpt,
                predict=True,
            )
            run_job(
                train_script,
                f"Training using pseudolabels (iter {idx})",
                ckpt,
                use_pseudo=1,
            )

        ckpt = get_best_ckpt(paths["checkpoint_dir"])
        print(f"Best checkpoint after all iterations: {ckpt}")

        print("Saving backbone model...")
        res = subprocess.run(
            [
                "uv",
                "run",
                "save_model.py",
                "--ckpt",
                str(ckpt),
                "--save_to",
                str(paths["model_path"]),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Failed to save model: {res.stderr}")
        print(res.stdout.strip())

        run_job(fusion_script, "Training fusion model", paths["model_path"])

        fusion_ckpt = get_best_ckpt(paths["fusion_checkpoint_dir"])
        print(f"Best fusion checkpoint: {fusion_ckpt}")

        print("Saving fusion model...")
        res = subprocess.run(
            [
                "uv",
                "run",
                "save_model.py",
                "--ckpt",
                str(fusion_ckpt),
                "--save_to",
                str(paths["fusion_model_path"]),
                "--type",
                "f",
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Failed to save fusion model: {res.stderr}")
        print(res.stdout.strip())

        run_job(
            predict_fusion_script,
            "Predicting final labels with fusion model",
            fusion_ckpt,
            predict=True,
            extras=[paths["fusion_model_path"]],
        )

        print("All jobs completed successfully.")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


def run_predict_only(config: Dict[str, Any], paths: Dict[str, Path]):
    """Run only the prediction step using the best checkpoint"""
    print("\n🏃 Starting prediction pipeline...")

    # Lock scripts at startup
    # locked_scripts_dir = lock_scripts(config, paths)
    SCRIPTS_DIR = paths["scripts_backup_dir"]

    predict_script = SCRIPTS_DIR / "predict_final.sh"

    if not predict_script.exists():
        raise FileNotFoundError(f"Predict script not found: {predict_script}")

    print(f"Using script: {predict_script}")

    ckpt = get_best_ckpt(paths["fusion_checkpoint_dir"])
    print(f"Using checkpoint for prediction: {ckpt}")

    run_job(
        predict_script,
        "Predicting final labels",
        ckpt,
        predict=True,
        extras=[paths["fusion_model_path"]],
    )

    print("Prediction completed successfully.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Centralized Experiment Runner")
    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="Whether to run in prediction mode only",
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
        help="Model type (convnext or medvit)",
        choices=["convnext", "medvit", "vit"],
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
        "--image_size", type=int, help="Image size for training", required=True
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
    )
    parser.add_argument(
        "--predict_type",
        type=str,
        help="Prediction type (dev or test)",
        default="dev",
        choices=["dev", "test"],
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

    # Override config with command-line arguments if provided
    for key, value in vars(args).items():
        if value is not None and key != "predict_only":
            config[key] = value

    predict_type = config.get("predict_type", "dev")

    pred_df_dir = Path("data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/")
    if predict_type == "dev":
        pred_df_path = pred_df_dir / f"development_labeled_{config['task']}.csv"
    else:
        pred_df_path = pred_df_dir / f"test_labeled_{config['task']}.csv"

    config["pred_df_path"] = str(pred_df_path)

    paths = get_experiment_paths(config, "docker")

    print("=" * 60)
    print("🎯 CENTRALIZED EXPERIMENT RUNNER")
    print("=" * 60)

    # Setup experiment
    setup_experiment(config, paths)  # type: ignore

    if args.predict_only:
        run_predict_only(config, paths)  # type: ignore
        return

    # Run the full pipeline
    print("\n🏃 Starting training pipeline...")
    run_full_pipeline(config, paths)  # type: ignore

    print("\n🎉 Experiment completed successfully!")
    print(f"📁 Results saved to: {paths['submission_dir']}")
    print(
        f"🎯 Confusion matrix: {paths['submission_dir']}/conf_matrix_lr{config['lr']}.png"
    )


if __name__ == "__main__":
    main()
