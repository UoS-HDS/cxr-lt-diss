"""
Experiment utilities package
"""

from typing import Sequence, Generator

import timm
import torch
from torch import Tensor
import torch.utils.checkpoint as checkpoint
from torch.nn import Module
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertTokenizerFast,
)

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

    if (
        model_name in ["convnext", "convnextv2", "vit", "maxvit"]
        and init_args is not None
    ):
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


def get_embedding_model(model_name: str) -> tuple[BertTokenizerFast, BertModel]:
    """
    Get label embeddings from an NLP model
    Args:
        model_name (str): Name of the model to load embeddings from
                          options: PubMedBERT, UmlsBert
    Returns:
        embeddings (torch.Tensor): Tensor of shape [num_labels, embedding_dim]
    """
    if model_name.lower() == "pubmedbert":
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        )
        model = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        )
    elif model_name.lower() == "umlsbert":
        tokenizer = AutoTokenizer.from_pretrained("checkpoints/umlsbert/")
        model = AutoModel.from_pretrained("checkpoints/umlsbert/")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return tokenizer, model


def get_label_embeddings(
    model_name: str,
    labels: list[str],
) -> Tensor:
    """
    Get label embeddings for a given text using the specified model and tokenizer.
    Args:
        model_name (str): The name of the NLP model to use for generating embeddings.
        labels (list[str]): The list of labels to generate embeddings for.
    Returns:
        Tensor: The word embeddings for the input text.
    """
    tokenizer, model = get_embedding_model(model_name)
    embeddings = []
    for label in labels:
        tokens = tokenizer(label, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings.append(outputs.pooler_output.squeeze(0))
    return torch.stack(embeddings)  # Shape: [num_labels, hidden_size]
