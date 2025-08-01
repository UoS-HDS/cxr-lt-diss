#!/usr/bin/env python3
"""
Centralized Experiment Runner for Apptainer/SLURM
Single point of control for all experiments using Apptainer containers

Based on the original train_full_docker2.py but with Apptainer/SLURM variations.
Edit src/utils/experiment_config.py to change experiment settings.
"""

import subprocess
import signal
import sys
import os
import re
import time
from pathlib import Path
from typing import Dict, Any
import argparse

# Import our utilities
from src.utils.experiment_config import (
    CURRENT_EXPERIMENT,
    get_experiment_paths,
)
from src.utils.config_generator import write_config_files
from src.utils.script_generator_apptainer import write_apptainer_script_files
from src.utils.directory_manager import create_experiment_directories


# Global variable to track running job IDs
running_job_ids = []


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully by cancelling all running SLURM jobs."""
    print("\nInterrupt received. Cancelling all running SLURM jobs...")
    for job_id in running_job_ids:
        try:
            subprocess.run(["scancel", job_id], check=False)
            print(f"Cancelled job {job_id}")
        except Exception as e:
            print(f"Failed to cancel job {job_id}: {e}")
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


def submit_job(
    script_path: Path,
    ckpt_path: Path | None = None,
    use_pseudo: int = 0,
    predict: bool = False,
) -> str:
    """
    Submit a SLURM job using sbatch and return the job ID.
    """
    # Make script executable
    os.chmod(script_path, 0o755)

    if predict:
        if ckpt_path:
            cmd = ["sbatch", str(script_path), str(ckpt_path)]
        else:
            cmd = ["sbatch", str(script_path), "null"]
    else:
        if ckpt_path and use_pseudo:
            cmd = ["sbatch", str(script_path), str(use_pseudo), str(ckpt_path)]
        elif ckpt_path:
            cmd = ["sbatch", str(script_path), str(ckpt_path)]
        else:
            cmd = ["sbatch", str(script_path), str(use_pseudo)]

    print(f"Running command: {' '.join(cmd)}")

    # Submit job to get job ID
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    # Extract job ID from output
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise ValueError(
            f"Could not extract job ID from sbatch output: {result.stdout}"
        )
    job_id = match.group(1)
    print(f"Submitted job {job_id}")

    # Add to tracking list immediately
    running_job_ids.append(job_id)

    return job_id


def poll_job_status(job_id: str) -> str:
    """Poll the job status using squeue and sacct."""
    result = subprocess.run(
        ["squeue", "--job", job_id, "--format=%T", "--noheader"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not result.stdout.strip():
        # Job might have finished, check with sacct
        sacct_result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
        )
        if sacct_result.returncode == 0 and sacct_result.stdout.strip():
            states = [
                s.strip() for s in sacct_result.stdout.strip().split("\n") if s.strip()
            ]
            if any(states):
                return states[0]
        return "UNKNOWN"

    return result.stdout.strip()


def get_output_file_path(job_id: str) -> str:
    """Retrieve the output file path using scontrol."""
    scontrol_result = subprocess.run(
        ["scontrol", "show", "job", job_id],
        capture_output=True,
        text=True,
    )

    if scontrol_result.returncode == 0:
        for line in scontrol_result.stdout.split("\n"):
            if "StdOut=" in line:
                stdout_match = re.search(r"StdOut=(\S+)", line)
                if stdout_match:
                    return stdout_match.group(1)

    # Fallback to standard naming
    return f"slurm-{job_id}.out"


def tail_output_file(output_file: str, job_id: str) -> None:
    """Read the output file in real-time until the job finishes."""
    print(f"Monitoring output file: {output_file}")

    # Wait for file to be created
    while not os.path.exists(output_file):
        time.sleep(1)

    try:
        with open(output_file, "r") as f:
            while True:
                # Check job status first
                job_state = poll_job_status(job_id)
                if job_state not in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]:
                    print(f"Job {job_id} finished with state: {job_state}.")
                    # Read any remaining content
                    remaining = f.read()
                    if remaining:
                        print(remaining.rstrip())
                    break

                # Read new content
                line = f.readline()
                if line:
                    print(line.rstrip())
                else:
                    # No new content, sleep briefly
                    time.sleep(0.5)

    except FileNotFoundError:
        print(f"Output file {output_file} not found")
    except Exception as e:
        print(f"Error while reading output file: {e}")
        raise


def finalize_job(job_id: str, job_name: str) -> None:
    """Finalize the job by checking its final state."""
    final_state = poll_job_status(job_id)
    if final_state == "COMPLETED":
        print(f"{job_name} completed successfully (Job ID: {job_id}).")
    else:
        raise RuntimeError(
            f"{job_name} finished with state: {final_state} (Job ID: {job_id})"
        )


def wait_for_job_completion(job_id: str, job_name: str):
    """Wait for a SLURM job to complete and stream its output."""
    print(f"Starting {job_name}...")

    # Wait for the job to start running
    while True:
        job_state = poll_job_status(job_id)
        if job_state == "RUNNING":
            print(f"Job {job_id} is now running. Monitoring output...")
            break
        elif job_state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
            print(f"Job {job_id} finished with state: {job_state} before running.")
            finalize_job(job_id, job_name)
            return
        else:
            print(f"Job {job_id} status: {job_state}. Waiting...")
            time.sleep(10)

    # Get the output file path and tail it
    output_file = get_output_file_path(job_id)
    tail_output_file(output_file, job_id)

    # Finalize the job
    finalize_job(job_id, job_name)


def run_job(
    script_path: Path,
    job_name: str,
    ckpt_path: Path | None = None,
    use_pseudo: int = 0,
    predict: bool = False,
):
    """
    Run a SLURM job and wait for completion, with proper job tracking for cancellation.
    """
    job_id = submit_job(
        script_path=script_path,
        ckpt_path=ckpt_path,
        use_pseudo=use_pseudo,
        predict=predict,
    )

    wait_for_job_completion(job_id, job_name)


def validate_apptainer_config(config: Dict[str, Any]) -> None:
    """Validate experiment configuration for Apptainer"""
    required_keys = [
        "task",
        "model_type",
        "model_name",
        "loss_type",
        "image_size",
        "lr",
        "batch_size",
        "gpu_count",
        "apptainer_image",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if config["model_type"] not in ["convnext", "medvit", "vit", "maxvit", "random"]:
        raise ValueError(f"Invalid model_type: {config['model_type']}")

    if config["loss_type"] not in ["asl", "ral"]:
        raise ValueError(f"Invalid loss_type: {config['loss_type']}")

    if config["task"] not in ["task1", "task2", "task3"]:
        raise ValueError(f"Invalid task: {config['task']}")

    if not isinstance(config["gpu_count"], int) or config["gpu_count"] <= 0:
        raise ValueError("gpu_count must be a positive integer")


def setup_experiment(config: Dict[str, Any], paths: Dict[str, Path]):
    """Set up all experiment files and directories"""

    exp_name = paths["exp_name"]
    print(f"🚀 Setting up experiment: {exp_name}")
    print(
        f"📊 Model: {config['model_type']} | Loss: {config['loss_type']} | Size: {config['image_size']} | GPUs: {config['gpu_count']}"
    )

    # 1. Validate configuration
    validate_apptainer_config(config)

    # 2. Create directories
    create_experiment_directories(paths)

    # 3. Generate configuration files
    print("\n📝 Generating configuration files...")
    write_config_files(config, paths)

    # 4. Generate Apptainer script files
    print("\n🔧 Generating Apptainer scripts...")
    write_apptainer_script_files(config, paths)

    print("\n✅ Experiment setup complete!")


def run_full_pipeline(
    config: Dict[str, Any],
    paths: Dict[str, Path],
    fusion_only: bool = False,
):
    """Run the complete training pipeline"""

    # Use generated scripts
    SCRIPTS_DIR = paths["scripts_backup_dir"]

    # Define script paths
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
            if fusion_only:
                continue

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

        if not fusion_only:
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
        )

        print("All jobs completed successfully.")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


def run_predict_only(config: Dict[str, Any], paths: Dict[str, Path]):
    """Run only the prediction step using the best checkpoint"""
    print("\n🏃 Starting prediction pipeline...")

    # Use generated scripts
    SCRIPTS_DIR = paths["scripts_backup_dir"]

    predict_script = SCRIPTS_DIR / "predict_final.sh"

    if not predict_script.exists():
        raise FileNotFoundError(f"Predict script not found: {predict_script}")

    print(f"Using script: {predict_script}")

    if config["model_type"] not in ["random"]:
        fusion_ckpt = get_best_ckpt(paths["fusion_checkpoint_dir"])
        print(f"Using checkpoint for prediction: {fusion_ckpt}")

        run_job(
            predict_script,
            "Predicting final labels",
            fusion_ckpt,
            predict=True,
        )
    else:
        run_job(
            predict_script,
            "Predicting final labels with random model",
            None,
            predict=True,
        )

    print("Prediction completed successfully.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Centralized Experiment Runner for Apptainer/SLURM"
    )
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
        choices=["convnext", "medvit", "vit", "maxvit", "random"],
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
        "--gpu_count", type=int, help="Number of GPUs for training", required=True
    )
    parser.add_argument("--mem", type=str, help="Memory for SLURM jobs", default="48G")
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
        "--project_dir",
        type=str,
        help="Project directory path",
        default="/mnt/isilon1/na200/hds-diss",
    )
    parser.add_argument(
        "--apptainer_image",
        type=str,
        help="Apptainer image name",
        required=True,
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="SLURM partition to use",
        default="gpu",
    )
    parser.add_argument(
        "--time",
        type=str,
        help="Time limit for SLURM jobs",
        default="72:00:00",
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

    pred_df_dir = Path("data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/")
    if predict_type == "dev":
        pred_df_path = pred_df_dir / f"development_labeled_{config['task']}.csv"
    else:
        pred_df_path = pred_df_dir / f"test_labeled_{config['task']}.csv"

    config["pred_df_path"] = str(pred_df_path)
    paths = get_experiment_paths(config, "apptainer")

    print("=" * 60)
    print("🎯 CENTRALIZED EXPERIMENT RUNNER (APPTAINER/SLURM)")
    print("=" * 60)

    # Setup experiment
    setup_experiment(config, paths)  # type: ignore

    if args.predict_only:
        run_predict_only(config, paths)  # type: ignore
        return

    # Run the full pipeline
    print("\n🏃 Starting training pipeline...")
    run_full_pipeline(config, paths, args.fusion_only)  # type: ignore

    print("\n🎉 Experiment completed successfully!")
    print(f"📁 Results saved to: {paths['submission_dir']}")
    print(f"🎯 Confusion matrix: {paths['conf_matrix_path']}")


if __name__ == "__main__":
    main()
