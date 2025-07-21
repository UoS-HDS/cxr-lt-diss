"""
saves models
"""

from pathlib import Path
from typing import Literal

import torch
import argparse

from src.models.chexfusion import CxrModel, CxrModelFusion


PARSER = argparse.ArgumentParser(
    usage="python (or `uv run`) save_backbone.py --type <model_type> --ckpt <checkpoint_path> --fname <filename>",
    description="Save the backbone from a trained model checkpoint.",
)
PARSER.add_argument(
    "--ckpt",
    type=str,
    required=True,
    help="Path to the model checkpoint file.",
)
PARSER.add_argument(
    "--save_to",
    type=str,
    required=True,
    help="path to save the model state dictionary, including the filename + ext.",
)
PARSER.add_argument(
    "--create_parent",
    type=bool,
    default=False,
    help="If True, create the parent directory if it does not exist.",
)
PARSER.add_argument(
    "--type",
    type=str,
    choices=["b", "f"],
    default="b",
    help="Type of model to save. 'b' for backbone, 'f' for fusion model.",
)


def save_model(
    type: Literal["b", "f"],
    ckpt_path: str,
    save_to: str,
    create_parent: bool = False,
):
    if type == "b":
        model = CxrModel.load_from_checkpoint(checkpoint_path=ckpt_path)
    elif type == "f":
        model = CxrModelFusion.load_from_checkpoint(checkpoint_path=ckpt_path)
    else:
        raise ValueError(
            f"Unknown model type: {type}. Use 'b' for backbone or 'f' for fusion model."
        )
    path = Path(save_to)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.exists():
        raise ValueError(
            f"Invalid save path: {path.parent}. Please provide a valid path."
        )

    torch.save(model.backbone.model.state_dict(), path)


if __name__ == "__main__":
    args = PARSER.parse_args()
    save_model(args.type, args.ckpt, args.save_to, args.create_parent)
    print(f"Model saved to {args.save_to}")
