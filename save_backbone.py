"""
saves the backbone from stage 1 to be used frozen in stage 2
"""

from pathlib import Path

import torch
import yaml
from src.models.chexfusion import CxrModel

BACKBONE_DIR = Path("checkpoints/backbones")
BACKBONE_DIR.mkdir(parents=True, exist_ok=True)


def save_backbone(config_path: str | Path, fname: str | Path = "model.pth"):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # print(config.get("model"))
        model = CxrModel.load_from_checkpoint(config["ckpt_path"])

        torch.save(model.backbone.model.state_dict(), BACKBONE_DIR / fname)
