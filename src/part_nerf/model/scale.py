from functools import partial
from typing import Dict

import torch
from torch import nn

from .utils import linear_block


class ScaleFromEmbedding(nn.Module):
    def __init__(self, in_dims: int, min_a: float = 0.005, max_a: float = 0.5):
        super().__init__()
        self.min_a = min_a
        self.max_a = max_a
        self.fc = linear_block(in_dims, 3)

    def forward(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        primitive_features = X["part_shape_features"]
        scale = torch.sigmoid(self.fc(primitive_features)) * self.max_a + self.min_a
        X["scale"] = scale
        return X


class DeepScaleFromEmbedding(ScaleFromEmbedding):
    def __init__(
        self,
        in_dims: int,
        min_a: float = 0.005,
        max_a: float = 0.5,
        hidden_dim: int = 256,
        activation: str = "relu",
        normalization: str = None,
    ):
        super().__init__(in_dims=in_dims, min_a=min_a, max_a=max_a)
        self.fc = nn.Sequential(
            linear_block(
                in_channels=in_dims,
                out_channels=hidden_dim,
                activation=activation,
                normalization=normalization,
            ),
            linear_block(hidden_dim, 3),
        )


def get_scale_instance(name: str, config: Dict) -> nn.Module:
    feature_size = config["shape_decomposition_network"].get("output_size")
    activation = config.get("activation", "relu")
    normalization = config.get("normalization", None)
    min_a = config["structure_network"].get("scale_min_a", 0.005)
    max_a = config["structure_network"].get("scale_max_a", 0.5)
    scale_factory = {
        "embedding": partial(
            ScaleFromEmbedding,
            in_dims=feature_size,
            min_a=min_a,
            max_a=max_a,
        ),
        "embedding_deep": partial(
            DeepScaleFromEmbedding,
            in_dims=feature_size,
            min_a=min_a,
            max_a=max_a,
            hidden_dim=config.get("scale_hidden_dim", 256),
            activation=activation,
            normalization=normalization,
        ),
    }
    return scale_factory[name]()
