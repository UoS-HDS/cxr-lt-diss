"""
Stages/Layers for chexFusion. Adapted from
https://github.com/dongkyunk/CheXFusion/blob/main/model/layers.py
"""

from pathlib import Path

import torch
import timm
import torch.nn as nn
import copy
import einops
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from src.models.ml_decoder import MLDecoder


class Backbone(nn.Module):
    """Backbone trained in Stage 1"""

    def __init__(self, timm_init_args: dict):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.model.head = nn.Identity()
        self.pos_encoding = Summer(PositionalEncoding2D(768))
        self.head = MLDecoder(
            num_classes=timm_init_args["num_classes"],
            initial_num_features=768,
        )

    def forward(self, x):
        x = self.model(x)
        x = self.pos_encoding(x)
        x = self.head(x)
        return x


class FusionBackbone(nn.Module):
    """
    Fusion backbone for Stage 2

    Provide `pretrained_path` to load pretrained weights (freeze the CNN backbone).
    If pretrained_path is None, the model will be initialized randomly and trained
    """

    def __init__(self, timm_init_args: dict, pretrained_path: str | Path | None = None):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.model.head = nn.Identity()
        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path))
        self.model.head = nn.Identity()
        self.conv2d = nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = Summer(PositionalEncoding2D(768))
        self.padding_token = nn.Parameter(torch.randn(1, 768, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(4, 768, 1, 1))

        self.head = MLDecoder(
            num_classes=timm_init_args["num_classes"],
            initial_num_features=768,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=2,
        )

    def forward(self, x):
        b, s, _, _, _ = x.shape

        x = einops.rearrange(x, "b s c h w -> (b s) c h w")
        no_pad = torch.nonzero(x.sum(dim=(1, 2, 3)) != 0).squeeze(1)
        x = x[no_pad]

        with torch.no_grad():
            x = self.model(x).detach()

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
