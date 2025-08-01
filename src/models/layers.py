"""
Stages/Layers for chexFusion. Adapted from
https://github.com/dongkyunk/CheXFusion/blob/main/model/layers.py
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import timm
import torch.nn as nn
import copy
import einops

# from torch.utils.checkpoint import checkpoint
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from src.models.ml_decoder import MLDecoder
from src.utils import get_model, get_label_embeddings


class MajorityClassClassifier(nn.Module):
    pass


class RandomClassifier(nn.Module):
    """Random classifier for baseline
    uses the class proportions in the training set
    """

    def __init__(
        self,
        model_init_args: dict[str, Any],
        loss_init_args: dict[str, Any],
        classes: list[str],
    ):
        super().__init__()
        self.classes = classes
        self.class_nums = np.array(loss_init_args["class_instance_nums"])
        self.total_images = np.array(loss_init_args["total_instance_num"])
        self.class_props = self.class_nums / self.total_images
        self.model = nn.Linear(1, len(classes), bias=False)
        self.model.weight = nn.Parameter(
            torch.tensor(self.class_props, dtype=torch.float32).unsqueeze(0),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is not used, but required for compatibility
        # with the training loop
        return self.model(x.unsqueeze(1)).squeeze(1)


class Backbone(nn.Module):
    """Backbone trained in Stage 1"""

    def __init__(
        self,
        model_type: str,
        model_init_args: dict[str, Any],
        classes: list[str],
        embedding: str | None = None,
        zsl: int = 0,
        target_dim: int = 768,
    ):
        super().__init__()
        self.model = get_model(model_type, model_init_args)
        self.model.head = nn.Identity()

        # Auto-detect input dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 1024, 1024)  # Use your actual input size
            dummy_output = self.model(dummy_input)
            input_dim = dummy_output.shape[1]

        # Project to standard dimension for fair comparison
        if input_dim != target_dim:
            self.projection = nn.Conv2d(
                input_dim, target_dim, kernel_size=1, bias=False
            )
        else:
            self.projection = nn.Identity()

        self.pos_encoding = Summer(PositionalEncoding2D(target_dim))
        if embedding is not None:
            embeddings = get_label_embeddings(embedding, classes)
            self.head = MLDecoder(
                num_classes=model_init_args["num_classes"],
                initial_num_features=target_dim,
                zsl=zsl,
                embeddings=embeddings,
            )
        else:
            self.head = MLDecoder(
                num_classes=model_init_args["num_classes"],
                initial_num_features=target_dim,
                zsl=zsl,
            )

    def forward(self, x):
        x = self.model(x)

        x = self.projection(x)  # Project to standard 768 dimensions
        x = self.pos_encoding(x)
        x = self.head(x)
        return x


class FusionBackbone(nn.Module):
    """
    Fusion backbone for Stage 2

    Provide `pretrained_path` to load pretrained weights (freeze the CNN backbone).
    If pretrained_path is None, the model will be initialized randomly and trained
    """

    def __init__(
        self,
        model_type: str,
        model_init_args: dict[str, Any],
        classes: list[str],
        embedding: str | None = None,
        zsl: int = 0,
        target_dim: int = 768,
        pretrained_path: str | Path | None = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.model = get_model(model_type, model_init_args)
        self.model.head = nn.Identity()

        # Auto-detect input dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 1024, 1024)
            dummy_output = self.model(dummy_input)
            input_dim = dummy_output.shape[1]

        if pretrained_path is not None:
            print(f"Using pretrained backbone: {pretrained_path}")
            self.model.load_state_dict(torch.load(pretrained_path))
        self.model.head = nn.Identity()

        # Project backbone features to standard dimension
        if input_dim != target_dim:
            self.backbone_projection = nn.Conv2d(
                input_dim, target_dim, kernel_size=1, bias=False
            )
        else:
            self.backbone_projection = nn.Identity()

        self.conv2d = nn.Conv2d(
            target_dim, target_dim, kernel_size=3, stride=2, padding=1
        )
        self.pos_encoding = Summer(PositionalEncoding2D(target_dim))
        self.padding_token = nn.Parameter(torch.randn(1, target_dim, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(4, target_dim, 1, 1))

        if embedding is not None:
            embeddings = get_label_embeddings(embedding, classes)
            self.head = MLDecoder(
                num_classes=model_init_args["num_classes"],
                initial_num_features=target_dim,
                zsl=zsl,
                embeddings=embeddings,
            )
        else:
            self.head = MLDecoder(
                num_classes=model_init_args["num_classes"],
                initial_num_features=target_dim,
                zsl=zsl,
            )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=target_dim, nhead=8),
            num_layers=2,
        )

    def forward(self, x):
        b, s, _, _, _ = x.shape

        x = einops.rearrange(x, "b s c h w -> (b s) c h w")
        no_pad = torch.nonzero(x.sum(dim=(1, 2, 3)) != 0).squeeze(1)
        x = x[no_pad]

        if self.freeze_backbone:
            with torch.no_grad():
                x = self.model(x).detach()
        else:
            x = self.model(x)

        x = self.backbone_projection(x)  # Project to standard dimensions
        x = self.conv2d(x)
        x = self.pos_encoding(x)

        pad_tokens = einops.repeat(
            self.padding_token,
            "1 c 1 1 -> (b s) c h w",
            b=b,
            s=s,
            h=x.shape[2],
            w=x.shape[3],
        ).type_as(x)
        segment_embedding = einops.repeat(
            self.segment_embedding,
            "s c 1 1 -> (b s) c h w",
            b=b,
            h=x.shape[2],
            w=x.shape[3],
        ).type_as(x)
        pad_tokens[no_pad] = x + segment_embedding[no_pad]
        x = pad_tokens

        x = einops.rearrange(
            x,
            "(b s) c h w -> b (s h w) c",
            b=b,
            s=s,
            h=x.shape[2],
            w=x.shape[3],
        )
        mask = (x.sum(dim=-1) == 0).transpose(0, 1)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.head(x, mask)

        return x


class PretrainedBackbone(nn.Module):
    def __init__(self, timm_init_args: dict, pretrained_path):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.new_head = copy.deepcopy(self.model.head)
        self.model.load_state_dict(torch.load(pretrained_path))
        self.model.head = nn.Identity()

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        x = self.new_head(x.detach())
        return x
