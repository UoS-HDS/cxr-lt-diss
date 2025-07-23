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
    extras: list[str | Path] = [],
) -> str:
    """
    Submit a SLURM job using sbatch and return the job ID.
    """
    # Make script executable
    os.chmod(script_path, 0o755)

    if not ckpt_path:
        cmd = ["sbatch", str(script_path), str(use_pseudo)]
    else:
        if predict and extras:
            cmd = [
                "sbatch",
                str(script_path),
                str(ckpt_path),
                *[str(x) for x in extras],
            ]
        elif predict:
            cmd = ["sbatch", str(script_path), str(ckpt_path)]
        elif not use_pseudo:
            cmd = ["sbatch", str(script_path), str(ckpt_path)]
        else:
            cmd = [
                "sbatch",
                str(script_path),
                str(use_pseudo),
                str(ckpt_path),
            ]

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


def wait_for_job_completion(job_id: str, job_name: str):
    """
    Wait for a SLURM job to start running, then stream its output using sattach.
    """
    print(f"Starting {job_name}...")

    # Poll until job starts running
    print(f"Waiting for job {job_id} to start running...")
    job_state = "PENDING"  # Initialize job_state
    while True:
        result = subprocess.run(
            ["squeue", "--job", job_id, "--format=%T", "--noheader"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Job might have finished already, check with sacct
            sacct_result = subprocess.run(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                capture_output=True,
                text=True,
            )
            if sacct_result.returncode == 0 and sacct_result.stdout.strip():
                states = [
                    s.strip()
                    for s in sacct_result.stdout.strip().split("\n")
                    if s.strip()
                ]
                if states:
                    final_state = states[0]
                    if final_state in ["COMPLETED"]:
                        print(
                            f"{job_name} completed before we could attach (Job ID: {job_id})."
                        )
                        return
                    else:
                        raise RuntimeError(
                            f"{job_name} failed with state: {final_state} (Job ID: {job_id})"
                        )
            break

        job_state = result.stdout.strip()
        if job_state in ["RUNNING"]:
            print(f"Job {job_id} is now running. Monitoring output...")
            break
        elif job_state in ["PENDING", "CONFIGURING"]:
            print(f"Job {job_id} status: {job_state}. Waiting...")
            time.sleep(10)
        elif job_state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
            print(f"Job {job_id} finished with state: {job_state}")
            break
        else:
            print(f"Job {job_id} in unknown state: {job_state}. Continuing...")
            break

    # If job is running or finished, start tailing output files
    if job_state in ["RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
        # Get actual output file path from SLURM
        scontrol_result = subprocess.run(
            ["scontrol", "show", "job", job_id],
            capture_output=True,
            text=True,
        )

        output_file = None
        if scontrol_result.returncode == 0:
            # Parse the scontrol output to find StdOut path
            for line in scontrol_result.stdout.split("\n"):
                if "StdOut=" in line:
                    # Extract the path after StdOut=
                    stdout_match = re.search(r"StdOut=(\S+)", line)
                    if stdout_match:
                        output_file = stdout_match.group(1)
                        break

        if not output_file:
            # Fallback to standard naming if we can't get it from scontrol
            output_file = f"slurm-{job_id}.out"
            print(
                f"Could not determine output file from scontrol, using default: {output_file}"
            )
        else:
            print(f"Monitoring job {job_id} output file: {output_file}")

        # Tail the output file in real-time
        tail_process = None
        try:
            # Use tail -f to follow the output file
            tail_process = subprocess.Popen(
                ["tail", "-f", output_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Stream output from tail
            while True:
                # Check if job is still running
                result = subprocess.run(
                    ["squeue", "--job", job_id, "--noheader"],
                    capture_output=True,
                    text=True,
                )

                job_still_running = result.returncode == 0 and result.stdout.strip()

                # If job finished, read any remaining output and exit
                if not job_still_running:
                    print(f"Job {job_id} has finished. Reading final output...")
                    # Read any remaining output from tail
                    try:
                        if tail_process.stdout:
                            # Read all remaining lines
                            while True:
                                line = tail_process.stdout.readline()
                                if not line:
                                    break
                                print(line.rstrip())
                    except Exception:
                        pass
                    break

                # Read available output (only if job is still running)
                if tail_process.stdout:
                    try:
                        # Use select to check if data is available (non-blocking)
                        import select

                        if select.select([tail_process.stdout], [], [], 1)[0]:
                            line = tail_process.stdout.readline()
                            if line:
                                print(line.rstrip())
                    except Exception:
                        # Fallback if select doesn't work
                        pass

                time.sleep(1)

        except FileNotFoundError:
            print(
                f"Output file {output_file} not found. Monitoring job status instead..."
            )
            # Fallback: monitor job status without output
            while True:
                result = subprocess.run(
                    ["squeue", "--job", job_id, "--noheader"],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0 or not result.stdout.strip():
                    break

                print(f"Job {job_id} still running...")
                time.sleep(30)

        except Exception as e:
            print(f"Error tailing output file: {e}")
            # Fallback: monitor job status without output
            while True:
                result = subprocess.run(
                    ["squeue", "--job", job_id, "--noheader"],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0 or not result.stdout.strip():
                    break

                print(f"Job {job_id} still running...")
                time.sleep(30)

        finally:
            # Clean up tail process
            if tail_process and tail_process.poll() is None:
                tail_process.terminate()
                try:
                    tail_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    tail_process.kill()

    try:
        # Check final job status
        sacct_result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
        )

        if sacct_result.returncode == 0:
            states = [
                s.strip() for s in sacct_result.stdout.strip().split("\n") if s.strip()
            ]
            if states:
                final_state = states[0]
                if final_state in ["COMPLETED"]:
                    print(f"{job_name} completed successfully (Job ID: {job_id}).")
                    return
                else:
                    raise RuntimeError(
                        f"{job_name} failed with state: {final_state} (Job ID: {job_id})"
                    )

        # If we can't determine status, assume success
        print(f"{job_name} completed (Job ID: {job_id}).")

    finally:
        # Remove from tracking list
        if job_id in running_job_ids:
            running_job_ids.remove(job_id)


def run_job(
    script_path: Path,
    job_name: str,
    ckpt_path: Path | None = None,
    use_pseudo: int = 0,
    predict: bool = False,
    extras: list[str | Path] = [],
):
    """
    Run a SLURM job and wait for completion, with proper job tracking for cancellation.
    """
    job_id = submit_job(
        script_path=script_path,
        ckpt_path=ckpt_path,
        use_pseudo=use_pseudo,
        predict=predict,
        extras=extras,
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

    if config["model_type"] not in ["convnext", "medvit", "vit"]:
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


def run_full_pipeline(config: Dict[str, Any], paths: Dict[str, Path]):
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

    # Use generated scripts
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
    parser = argparse.ArgumentParser(
        description="Centralized Experiment Runner for Apptainer/SLURM"
    )
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
        help="Model type (convnext, medvit, or vit)",
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

    # Override config with command-line arguments if provided
    for key, value in vars(args).items():
        if value is not None and key != "predict_only":
            config[key] = value

    paths = get_experiment_paths(config, "apptainer")

    if args.predict_only:
        run_predict_only(config, paths)  # type: ignore
        return

    print("=" * 60)
    print("🎯 CENTRALIZED EXPERIMENT RUNNER (APPTAINER/SLURM)")
    print("=" * 60)

    # Setup experiment
    setup_experiment(config, paths)  # type: ignore

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
