"""
custome types
"""

from typing import NamedTuple

from torch import Tensor


class LossInitArgs(NamedTuple):
    type: str
    class_instance_nums: Tensor
    total_instance_num: int | Tensor
