"""
Configuration file generator for experiments
Dynamically generates all config files based on experiment settings
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import copy

from src.utils.class_counts import (
    CLASSES_26,
    CLASSES_40,
    INSTANCE_NUMS_26,
    INSTANCE_NUMS_40,
    TOTAL_IMAGES,
)


def generate_main_config(
    config: Dict[str, Any], paths: Dict[str, Path]
) -> Dict[str, Any]:
    """Generate the main config.yaml content"""

    exp_name = paths["exp_name"]
    task = config["task"]
    if task in ["task1", "task3"]:
        classes = CLASSES_40
        instance_nums = INSTANCE_NUMS_40
        train_csv = "train_labeled.csv"
        auxillary_train_csv = "train_expanded.csv"
        n_classes = 40
    else:
        classes = CLASSES_26
        instance_nums = INSTANCE_NUMS_26
        train_csv = "train_labeled_26.csv"
        auxillary_train_csv = "train_expanded_26.csv"
        n_classes = 26

    main_config = {
        "seed_everything": 8089,
        "trainer": {
            "accelerator": "auto",
            "strategy": "ddp_find_unused_parameters_true",
            "devices": "auto",
            "num_nodes": 1,
            "precision": "16-mixed",
            "logger": [
                {
                    "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                    "init_args": {
                        # "save_dir": f"{str(paths['tb_log_dir'])}/iter_{config['iter']}",  # TODO: REMOVE iter
                        "save_dir": str(paths['tb_log_dir']),
                        "name": f"{exp_name}/{task}/stage-1",
                    },
                }
            ],
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        # "dirpath": f"{str(paths['checkpoint_dir'])}/iter_{config['iter']}",  # TODO: REMOVE iter
                        "dirpath": str(paths['checkpoint_dir']),
                        "filename": "{epoch:02d}-{val_loss:.4f}-{val_ap:.5f}",
                        "save_top_k": 1,
                        "save_last": True,
                        "monitor": "val_ap",
                        "mode": "max",
                    },
                },
                {
                    "class_path": "lightning.pytorch.callbacks.early_stopping.EarlyStopping",
                    "init_args": {
                        "monitor": "val_ap",
                        "mode": "max",
                        "patience": 10,
                        "min_delta": 0.002,
                    },
                },
            ],
            "fast_dev_run": False,
            "overfit_batches": 0.0,
            "val_check_interval": 0.25,
            "num_sanity_val_steps": 2,
            "accumulate_grad_batches": 1,
            "gradient_clip_val": None,
            "deterministic": None,
            "benchmark": True,
            "max_epochs": config.get("max_epochs", 150),
        },
        "model": {
            "lr": config["lr"],
            "embedding": config["embedding"],
            "zsl": config["zsl"],
            "classes": classes,
            "loss_init_args": {
                "type": config["loss_type"],
                "class_instance_nums": instance_nums,
                "total_instance_num": TOTAL_IMAGES,
            },
            "model_type": config["model_type"],
            "model_init_args": {
                "num_classes": n_classes,
                "model_name": config["model_name"],
                "pretrained": True,
            },
        },
        "data": {
            "dataloader_init_args": {
                "batch_size": config["batch_size"],
                "num_workers": 8,
                "pin_memory": True,
                "persistent_workers": True,
            },
            "datamodule_cfg": {
                "predict_pseudo_label": None,
                "use_pseudo_label": False,
                "loader_type": "single",
                "data_dir": "data/",
                "resized_dir": "data-resized/",
                "train_df_path": f"data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/{train_csv}",
                "task1_df_path": "data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/development_labeled_task1.csv",
                "task2_df_path": "data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/development_labeled_task2.csv",
                "zero_shot_df_path": "data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/development_task3.csv",
                "vinbig_train_df_path": f"data/vinbig-cxr/{auxillary_train_csv}",
                "vinbig_pseudo_train_df_path": str(paths["vinbig_pseudo_path"]),
                "nih_train_df_path": f"data/nih-cxr/{auxillary_train_csv}",
                "nih_pseudo_train_df_path": str(paths["nih_pseudo_path"]),
                "chexpert_train_df_path": f"data/chexpert/CheXpert-v1.0/{auxillary_train_csv}",
                "chexpert_pseudo_train_df_path": str(paths["chexpert_pseudo_path"]),
                "pred_df_path": str(paths["pred_df_path"]),
                "val_split": 0.1,
                "seed": 8089,
                "size": config["image_size"],
                "classes": classes,
            },
        },
        "ckpt_path": None,
    }

    main_config["trainer"]["callbacks"].insert(
        1,
        {
            "class_path": "src.callbacks.chexpert_pseudo_callback.ChexpertWriter",
            "init_args": {
                "write_interval": "epoch",
                "chexpert_train_df_path": f"data/chexpert/CheXpert-v1.0/{auxillary_train_csv}",
                "chexpert_pseudo_train_df_path": str(paths["chexpert_pseudo_path"]),
                "num_classes": n_classes,
            },
        },
    )

    return main_config


def generate_stage2_config(
    config: Dict[str, Any], paths: Dict[str, Path]
) -> Dict[str, Any]:
    """Generate stage-2 (fusion) config"""
    base_config = generate_main_config(config, paths)

    exp_name = paths["exp_name"]
    task = config["task"]

    # Modify for stage 2
    base_config["trainer"]["logger"][0]["init_args"]["name"] = (
        f"{exp_name}/{task}/stage-2"
    )
    base_config["trainer"]["callbacks"][0]["init_args"]["dirpath"] = str(
        paths["fusion_checkpoint_dir"]
    )

    # Remove pseudo-label callback for stage 2 if it exists
    base_config["trainer"]["callbacks"] = [
        cb
        for cb in base_config["trainer"]["callbacks"]
        if "ChexpertWriter" not in cb.get("class_path", "")
    ]

    # Add fusion-specific settings
    base_config["data"]["datamodule_cfg"]["loader_type"] = "fusion"
    base_config["model"]["pretrained_path"] = str(paths["model_path"])

    return base_config


def generate_stage2_pred_config(
    config: Dict[str, Any], paths: Dict[str, Path]
) -> Dict[str, Any]:
    """Generate stage-2 prediction config"""
    base_config = generate_stage2_config(config, paths)
    task = config["task"]
    pred_file = "preds" if config["predict_type"] == "dev" else "preds_test"

    # Add prediction callback
    prediction_callback = {
        "class_path": f"src.callbacks.{task}_callback.{task.capitalize()}SubmissionWriter",
        "init_args": {
            "sample_submit_path": "data/sample_submission.csv",
            "submit_path": str(paths["submission_dir"] / f"{pred_file}.csv"),
            "submit_zip_path": str(paths["submission_dir"] / f"{pred_file}.zip"),
            "submit_code_dir": str(paths["submission_dir"] / "code"),
            "pred_df_path": str(paths["pred_df_path"]),
            "write_interval": "epoch",
        },
    }
    base_config["trainer"]["callbacks"].append(prediction_callback)
    base_config["model"]["skip_predict_metrics"] = False
    base_config["model"]["conf_matrix_path"] = str(paths["conf_matrix_path"])
    base_config["model"]["pretrained_path"] = None
    base_config["data"]["datamodule_cfg"]["pred_df_path"] = str(paths["pred_df_path"])
    if task == "task3":
        classes = ["Bulla", "Cardiomyopathy", "Hilum", "Osteopenia", "Scoliosis"]
        instance_nums = [0, 0, 0, 0, 0]
        n_classes = 5
        base_config["model"]["classes"] = classes
        base_config["model"]["loss_init_args"]["class_instance_nums"] = instance_nums
        base_config["model"]["model_init_args"]["num_classes"] = n_classes
        base_config["data"]["datamodule_cfg"]["classes"] = classes

    return base_config


def generate_auxiliary_configs(
    config: Dict[str, Any], paths: Dict[str, Path]
) -> Dict[str, Dict[str, Any]]:
    """Generate auxiliary dataset configs (VinBig, NIH)"""
    base_config = generate_main_config(config, paths)
    task = config["task"]

    if task in ["task1", "task3"]:
        auxillary_train_csv = "train_expanded.csv"
        n_classes = 40
    else:
        auxillary_train_csv = "train_expanded_26.csv"
        n_classes = 26

    # VinBig config
    vinbig_config = copy.deepcopy(base_config)
    try:
        vinbig_callback = next(
            cb
            for cb in vinbig_config["trainer"]["callbacks"]
            if "ChexpertWriter" in cb["class_path"]
        )
        vinbig_callback["class_path"] = (
            "src.callbacks.vinbig_pseudo_callback.VinBigWriter"
        )
        vinbig_callback["init_args"] = {
            "write_interval": "epoch",
            "vinbig_train_df_path": f"data/vinbig-cxr/{auxillary_train_csv}",
            "vinbig_pseudo_train_df_path": str(paths["vinbig_pseudo_path"]),
            "num_classes": n_classes,
        }
        vinbig_config["data"]["datamodule_cfg"]["predict_pseudo_label"] = "vinbig"
    except StopIteration:
        # If ChexpertWriter is not found, we are not in a pseudo-labeling run
        pass

    # NIH config
    nih_config = copy.deepcopy(base_config)
    try:
        nih_callback = next(
            cb
            for cb in nih_config["trainer"]["callbacks"]
            if "ChexpertWriter" in cb["class_path"]
        )
        nih_callback["class_path"] = "src.callbacks.nih_pseudo_callback.NIHWriter"
        nih_callback["init_args"] = {
            "write_interval": "epoch",
            "nih_train_df_path": f"data/nih-cxr/{auxillary_train_csv}",
            "nih_pseudo_train_df_path": str(paths["nih_pseudo_path"]),
            "num_classes": n_classes,
        }
        nih_config["data"]["datamodule_cfg"]["predict_pseudo_label"] = "nih"
    except StopIteration:
        pass

    return {"config-vinbig.yaml": vinbig_config, "config-nih.yaml": nih_config}


def write_config_files(config: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """Generate and write all configuration files"""

    # Create config directory
    config_dir = Path(paths["configs_dir"])
    config_dir.mkdir(parents=True, exist_ok=True)

    # Generate all configs
    configs = {
        "config.yaml": generate_main_config(config, paths),
        "config-stage-2.yaml": generate_stage2_config(config, paths),
        "config-stage-2-pred.yaml": generate_stage2_pred_config(config, paths),
    }

    # Add auxiliary configs
    configs.update(generate_auxiliary_configs(config, paths))

    # Write all config files
    for filename, config_content in configs.items():
        config_path = config_dir / filename
        with open(config_path, "w") as f:
            yaml.dump(config_content, f, default_flow_style=False, sort_keys=False)

        print(f"✅ Generated: {config_path}")
