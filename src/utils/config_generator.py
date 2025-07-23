"""
Configuration file generator for experiments
Dynamically generates all config files based on experiment settings
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import copy


def generate_main_config(
    config: Dict[str, Any], paths: Dict[str, Path]
) -> Dict[str, Any]:
    """Generate the main config.yaml content"""

    exp_name = paths["exp_name"]

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
                        "save_dir": str(paths["tb_log_dir"]),
                        "name": f"{exp_name}/stage-1",
                    },
                }
            ],
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "dirpath": str(paths["checkpoint_dir"]),
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
                        "patience": 5,
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
            "classes": [
                "Adenopathy",
                "Atelectasis",
                "Azygos Lobe",
                "Calcification of the Aorta",
                "Cardiomegaly",
                "Clavicle Fracture",
                "Consolidation",
                "Edema",
                "Emphysema",
                "Enlarged Cardiomediastinum",
                "Fibrosis",
                "Fissure",
                "Fracture",
                "Granuloma",
                "Hernia",
                "Hydropneumothorax",
                "Infarction",
                "Infiltration",
                "Kyphosis",
                "Lobar Atelectasis",
                "Lung Lesion",
                "Lung Opacity",
                "Mass",
                "Nodule",
                "Normal",
                "Pleural Effusion",
                "Pleural Other",
                "Pleural Thickening",
                "Pneumomediastinum",
                "Pneumonia",
                "Pneumoperitoneum",
                "Pneumothorax",
                "Pulmonary Embolism",
                "Pulmonary Hypertension",
                "Rib Fracture",
                "Round(ed) Atelectasis",
                "Subcutaneous Emphysema",
                "Support Devices",
                "Tortuous Aorta",
                "Tuberculosis",
            ],
            "loss_init_args": {
                "type": config["loss_type"],
                "class_instance_nums": [
                    3137,
                    63471,
                    191,
                    4161,
                    71794,
                    158,
                    14822,
                    36137,
                    3462,
                    28673,
                    1075,
                    2672,
                    11193,
                    2794,
                    3756,
                    633,
                    710,
                    9431,
                    686,
                    126,
                    2202,
                    74659,
                    4978,
                    7140,
                    32885,
                    64252,
                    580,
                    3124,
                    694,
                    43744,
                    505,
                    13562,
                    1565,
                    864,
                    8704,
                    153,
                    2019,
                    83899,
                    3078,
                    1929,
                ],
                "total_instance_num": 248137,
            },
            "model_type": config["model_type"],
            "model_init_args": {
                "num_classes": 40,
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
                "train_df_path": "data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/train_labeled_no_missing_views.csv",
                "task1_df_path": "data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/development_labeled_task1.csv",
                "task2_df_path": "data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/development_labeled_task2.csv",
                "zero_shot_df_path": "data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/development_task3.csv",
                "vinbig_train_df_path": "data/vinbig-cxr/train_expanded.csv",
                "vinbig_pseudo_train_df_path": str(paths["vinbig_pseudo_path"]),
                "nih_train_df_path": "data/nih-cxr/train_expanded.csv",
                "nih_pseudo_train_df_path": str(paths["nih_pseudo_path"]),
                "chexpert_train_df_path": "data/chexpert/CheXpert-v1.0/train_expanded.csv",
                "chexpert_pseudo_train_df_path": str(paths["chexpert_pseudo_path"]),
                "val_split": 0.1,
                "seed": 8089,
                "size": config["image_size"],
                "classes": [
                    "Adenopathy",
                    "Atelectasis",
                    "Azygos Lobe",
                    "Calcification of the Aorta",
                    "Cardiomegaly",
                    "Clavicle Fracture",
                    "Consolidation",
                    "Edema",
                    "Emphysema",
                    "Enlarged Cardiomediastinum",
                    "Fibrosis",
                    "Fissure",
                    "Fracture",
                    "Granuloma",
                    "Hernia",
                    "Hydropneumothorax",
                    "Infarction",
                    "Infiltration",
                    "Kyphosis",
                    "Lobar Atelectasis",
                    "Lung Lesion",
                    "Lung Opacity",
                    "Mass",
                    "Nodule",
                    "Normal",
                    "Pleural Effusion",
                    "Pleural Other",
                    "Pleural Thickening",
                    "Pneumomediastinum",
                    "Pneumonia",
                    "Pneumoperitoneum",
                    "Pneumothorax",
                    "Pulmonary Embolism",
                    "Pulmonary Hypertension",
                    "Rib Fracture",
                    "Round(ed) Atelectasis",
                    "Subcutaneous Emphysema",
                    "Support Devices",
                    "Tortuous Aorta",
                    "Tuberculosis",
                ],
            },
        },
        "ckpt_path": None,
    }

    if config["task"] == "task1":
        main_config["trainer"]["callbacks"].insert(
            1,
            {
                "class_path": "src.callbacks.chexpert_pseudo_callback.ChexpertWriter",
                "init_args": {
                    "write_interval": "epoch",
                    "chexpert_train_df_path": "data/chexpert/CheXpert-v1.0/train_expanded.csv",
                    "chexpert_pseudo_train_df_path": str(paths["chexpert_pseudo_path"]),
                    "num_classes": 40,
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

    # Modify for stage 2
    base_config["trainer"]["logger"][0]["init_args"]["name"] = f"{exp_name}/stage-2"
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
    predict_type = config.get("predict_type", "dev")

    predict_df_dir = Path("data/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/")
    if predict_type == "dev":
        predict_df_path = predict_df_dir / f"development_labeled_{task}.csv"
    else:
        predict_df_path = predict_df_dir / f"test_labeled_{task}.csv"

    # Add prediction callback
    prediction_callback = {
        "class_path": f"src.callbacks.{task}_callback.{task.capitalize()}SubmissionWriter",
        "init_args": {
            "sample_submit_path": "data/sample_submission.csv",
            "submit_path": str(paths["submission_dir"] / "preds.csv"),
            "submit_zip_path": str(paths["submission_dir"] / "preds.zip"),
            "submit_code_dir": str(paths["submission_dir"] / "code"),
            "pred_df_path": str(predict_df_path),
            "write_interval": "epoch",
        },
    }
    base_config["trainer"]["callbacks"].append(prediction_callback)
    base_config["model"]["skip_predict_metrics"] = False
    base_config["model"]["conf_matrix_path"] = str(paths["conf_matrix_path"])

    return base_config


def generate_auxiliary_configs(
    config: Dict[str, Any], paths: Dict[str, Path]
) -> Dict[str, Dict[str, Any]]:
    """Generate auxiliary dataset configs (VinBig, NIH)"""
    base_config = generate_main_config(config, paths)

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
            "vinbig_train_df_path": "data/vinbig-cxr/train_expanded.csv",
            "vinbig_pseudo_train_df_path": str(paths["vinbig_pseudo_path"]),
            "num_classes": 40,
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
            "nih_train_df_path": "data/nih-cxr/train_expanded.csv",
            "nih_pseudo_train_df_path": str(paths["nih_pseudo_path"]),
            "num_classes": 40,
        }
        nih_config["data"]["datamodule_cfg"]["predict_pseudo_label"] = "nih"
    except StopIteration:
        pass

    return {"config-vinbig.yaml": vinbig_config, "config-nih.yaml": nih_config}


def write_config_files(config: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """Generate and write all configuration files"""

    # Create config directory
    config_dir = Path(paths["config_backup_dir"])
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
