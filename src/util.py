from typing import Sequence, Generator

import timm
import torch
import torch.utils.checkpoint as checkpoint
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer

from src.medvit.medvit import MedViT_small


def flatten(x: Sequence) -> Generator:
    """Recursively flattens arbitrarily nested sequences"""
    for item in x:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def get_model(model_name: str, init_args: dict | None = None) -> Module:
    """Get a model from timm or transformers"""

    if model_name in ["convnext", "convnextv2", "vit"] and init_args is not None:
        return timm.create_model(**init_args)
    elif model_name == "medvit":
        ckpt = torch.load("checkpoints/MedViT_small_im1k.pth")
        model = MedViT_small(pretrained=True)
        model.load_state_dict(ckpt["model"])

        # Replace forward method to return intermediate representation
        def forward_features(self, x):
            x = self.stem(x)
            for idx, layer in enumerate(self.features):
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(layer, x)
                else:
                    x = layer(x)
            x = self.norm(x)
            # Return 4D tensor instead of flattening to 2D
            # Shape will be [batch_size, channels, height, width] e.g., [16, 1024, 12, 12]
            return x  # Don't flatten - keep spatial dimensions

        # Bind the new forward method to the model
        import types

        model.forward = types.MethodType(forward_features, model)

        return model
    else:
        raise ValueError(
            f"Unknown model name: {model_name} or invalid init_args: {init_args}"
        )
