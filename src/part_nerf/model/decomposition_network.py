from functools import partial
from typing import Dict, Optional

import torch
from torch import nn


class DecompositionNetwork(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_parts: int,
        encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.upscale_layer = nn.Linear(in_dim, num_parts * out_dim)
        self.encoder = encoder
        self._num_parts = num_parts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.upscale_layer(x).reshape(B, self._num_parts, -1)
        x = self.encoder(x)
        return x


def get_decomposition_network(
    name: str, config: Dict, encoder: Optional[nn.Module] = None
) -> nn.Module:
    num_parts = config.get("num_parts")
    embedding_size = config.get("embedding_size")
    decomposition_size = config.get("output_size")
    decomposition_encoder = nn.Identity() if encoder is None else encoder
    decomposition_factory = {
        "simple": partial(
            DecompositionNetwork,
            in_dim=embedding_size,
            out_dim=decomposition_size,
            num_parts=num_parts,
            encoder=decomposition_encoder,
        ),
    }
    return decomposition_factory[name]()
