from functools import partial
from typing import Dict

import torch
from torch import nn

from .utils import linear_block


class TranslationFromEmbedding(nn.Module):
    def __init__(self, in_dims: int):
        super().__init__()
        self.fc = linear_block(in_dims, 3)

    def forward(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        primitive_features = X["part_shape_features"]
        X["translations"] = self.fc(primitive_features)
        return X


class DeepTranslationFromEmbedding(TranslationFromEmbedding):
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
            linear_block(hidden_dim, 3),
        )


class TranslationFromEmbeddingSeparate(nn.Module):
    def __init__(self, in_dims: int, num_parts: int):
        super().__init__()
        self._in_dims = in_dims
        self._num_parts = num_parts
        # each embedding produces different translation vector
        self.fc = linear_block(num_parts * in_dims, num_parts * 3)
        # initialize bias to be spread out in [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        spread_out_bias = (1 * torch.rand((num_parts, 3)) - 0.5).reshape(num_parts * 3)
        self.fc[0].bias = nn.Parameter(spread_out_bias)

    def forward(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        primitive_features = X["part_shape_features"]
        B, M, D = primitive_features.shape
        assert M == self._num_parts
        assert D == self._in_dims
        X["translations"] = self.fc(primitive_features.reshape((B, M * D))).reshape(
            (B, M, -1)
        )
        return X


def get_translation_instance(name: str, config: Dict) -> nn.Module:
    feature_size = config["shape_decomposition_network"].get("output_size")
    num_parts = config["shape_decomposition_network"].get("num_parts")
    activation = config.get("activation", "relu")
    normalization = config.get("normalization", None)
    translation_factory = {
        "embedding": partial(TranslationFromEmbedding, in_dims=feature_size),
        "embedding_deep": partial(
            DeepTranslationFromEmbedding,
            in_dims=feature_size,
            hidden_dim=config.get("translation_hidden_dim", 256),
            activation=activation,
            normalization=normalization,
        ),
        "embedding_separate": partial(
            TranslationFromEmbeddingSeparate,
            in_dims=feature_size,
            num_parts=num_parts,
        ),
    }
    return translation_factory[name]()
