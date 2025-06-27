"""
Contains various loss functions. Adapted from
https://github.com/dongkyunk/CheXFusion/blob/main/model/loss.py
"""

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


def get_loss(
    type: Literal["bce", "wbce", "asl", "ral"],
    class_instance_nums: Tensor,
    total_instance_num: int,
) -> nn.Module:
    """
    Instantiates and returns a specified loss function
    """
    if type == "bce":
        return nn.BCEWithLogitsLoss()
    elif type == "wbce":
        return BCEWithClassWeights(class_instance_nums, total_instance_num)
    elif type == "asl":
        return ASLWithClassWeight(class_instance_nums, total_instance_num)
    elif type == "ral":
        return RobustASL()
    else:
        raise ValueError(f"Unknown loss type: {type}")


class BCEWithClassWeights(nn.Module):
    """
    Weighted BCE loss
    $$
    """

    def __init__(
        self,
        class_instance_nums: Tensor,
        total_instance_num: int | Tensor,
    ):
        super().__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / total_instance_num
        self.pos_weights = torch.exp(1 - p)
        self.neg_weights = torch.exp(p)

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        # https://www.cse.sc.edu/~songwang/document/cvpr21d.pdf (equation 4)
        weight = label * self.pos_weights.cuda() + (1 - label) * self.neg_weights.cuda()
        loss = nn.functional.binary_cross_entropy_with_logits(
            pred, label, weight=weight
        )
        return loss


class ASLWithClassWeight(nn.Module):
    """
    Weighted Asymmetric Loss (ASL)
    $$
    """

    def __init__(
        self,
        class_instance_nums: Tensor,
        total_instance_num: int,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
    ):
        super().__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / total_instance_num
        self.pos_weights = torch.exp(1 - p)
        self.neg_weights = torch.exp(p)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, pred, label):
        weight = label * self.pos_weights.cuda() + (1 - label) * self.neg_weights.cuda()

        # Calculating Probabilities
        xs_pos = torch.sigmoid(pred)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        los_pos = label * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - label) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        loss *= weight

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * label
            pt1 = xs_neg * (1 - label)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * label + self.gamma_neg * (1 - label)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


class RobustASL(nn.Module):
    """Robust Asymmetric Loss (RAL)"""
