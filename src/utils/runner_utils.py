"""
Utility functions for managing and running experiments.
"""

from pathlib import Path
from typing import Dict, Any, Callable
import subprocess

from src.utils.config_generator import write_config_files
from src.utils.script_generator_apptainer import write_apptainer_script_files
from src.utils.script_generator import write_script_files
from src.utils.directory_manager import create_experiment_directories


def get_best_ckpt(ckpt_dir: Path) -> Path | None:
    """
    Get the best checkpoint file (using val_ap) from the specified directory.
    """

    def best_ap_idx(ckpts: list[Path]) -> int:
        max_ap, max_idx = 0.0, 0
        for i, ckpt in enumerate(ckpts):
            val_ap = ckpt.stem.split("val_ap=")[1]
            if float(val_ap) >= max_ap:
                max_ap = float(val_ap)
                max_idx = i

        return max_idx

    ckpt_files = [
        ckpt for ckpt in ckpt_dir.glob("*.ckpt") if ckpt.stem.find("last") == -1
    ]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}.")

    best_ckpt = ckpt_files[best_ap_idx(ckpt_files)]

    return best_ckpt


def setup_experiment(
    config: Dict[str, Any],
    paths: Dict[str, Path],
    runtime: str,
    valid_func: Callable[[Dict[str, Any]], None],
):
    """Set up all experiment files and directories"""

    exp_name = paths["exp_name"]
    print(f"🚀 Setting up experiment: {exp_name}")
    print(
        f"📊 Model: {config['model_type']} | Loss: {config['loss_type']} | Size: {config['image_size']} | GPUs: {config['gpu_count']}"
    )

    # 1. Validate configuration
    valid_func(config)

    # 2. Create directories
    create_experiment_directories(paths)

    # 3. Generate configuration files
    print("\n📝 Generating configuration files...")
    write_config_files(config, paths)

    # 4. Generate Apptainer script files
    print("\n🔧 Generating Apptainer scripts...")
    if runtime == "apptainer":
        write_apptainer_script_files(config, paths)
    else:
        write_script_files(config, paths)

    print("\n✅ Experiment setup complete!")


def run_full_pipeline(
    config: Dict[str, Any],
    paths: Dict[str, Path],
    run_job: Callable[..., None],
    valid_func: Callable[[Dict[str, Any]], None],
    runtime: str,
    fusion_only: bool = False,
):
    """Run the complete training pipeline"""

    # Use generated scripts
    SCRIPTS_DIR = paths["scripts_dir"]

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
            config["iter"] = idx
            setup_experiment(config, paths, runtime, valid_func)

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


def run_predict_only(
    config: Dict[str, Any],
    paths: Dict[str, Path],
    run_job: Callable[..., None],
):
    """Run only the prediction step using the best checkpoint"""
    print("\n🏃 Starting prediction pipeline...")

    # Use generated scripts
    SCRIPTS_DIR = paths["scripts_dir"]

    predict_script = SCRIPTS_DIR / "predict_final.sh"

    if not predict_script.exists():
        raise FileNotFoundError(f"Predict script not found: {predict_script}")

    print(f"Using script: {predict_script}")

    if config["model_type"] in ["random"]:
        run_job(
            predict_script,
            "Predicting final labels with random model",
            None,
            predict=True,
        )
    else:
        fusion_ckpt = get_best_ckpt(paths["fusion_checkpoint_dir"])
        print(f"Using checkpoint for prediction: {fusion_ckpt}")

        run_job(
            predict_script,
            "Predicting final labels",
            fusion_ckpt,
            predict=True,
        )

    print("Prediction completed successfully.")
