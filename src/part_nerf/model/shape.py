from functools import partial
from typing import Dict

import torch
from torch import nn


class EllipsoidShape(nn.Module):
    """Use this module when we want to define ellipsoids."""

    def __init__(self, num_parts: int):
        super().__init__()
        self._num_parts = num_parts

    def forward(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = X["part_shape_features"]
        B = features.shape[0]
        shapes = features.new_ones((B, self._num_parts, 2))
        X["shape"] = shapes
        return X


def get_shape_instance(name: str, config: Dict) -> nn.Module:
    num_parts = config["shape_decomposition_network"].get("num_parts")
    shape_factory = {
        "ellipsoid": partial(EllipsoidShape, num_parts=num_parts),
    }
    return shape_factory[name]()
