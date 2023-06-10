from functools import partial
from typing import Dict

import torch
from torch import nn

from .utils import linear_block


class RotationFromEmbedding(nn.Module):
    def __init__(self, in_dims: int):
        super().__init__()
        self.fc = linear_block(in_dims, 4)

    def forward(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        primitive_features = X["part_shape_features"]
        quats = self.fc(primitive_features)
        # Apply an L2-normalization non-linearity to enforce the unit norm
        # constrain
        rotations = quats / torch.norm(quats, 2, -1, keepdim=True)
        X["rotations"] = rotations
        return X


class DeepRotationFromEmbedding(RotationFromEmbedding):
    def __init__(
        self,
        in_dims: int,
        hidden_dim: int = 256,
        activation: str = "relu",
        normalization: str = None,
    ):
        super().__init__(in_dims=in_dims)
        self.fc = nn.Sequential(
            linear_block(
                in_channels=in_dims,
                out_channels=hidden_dim,
                activation=activation,
                normalization=normalization,
            ),
            linear_block(hidden_dim, 4),
        )


def get_rotation_instance(name: str, config: Dict) -> nn.Module:
    feature_size = config["shape_decomposition_network"].get("output_size")
    activation = config.get("activation", "relu")
    normalization = config.get("normalization", None)
    rotation_factory = {
        "embedding": partial(RotationFromEmbedding, in_dims=feature_size),
        "embedding_deep": partial(
            DeepRotationFromEmbedding,
            in_dims=feature_size,
            hidden_dim=config.get("rotation_hidden_dim", 256),
            activation=activation,
            normalization=normalization,
        ),
    }
    return rotation_factory[name]()
