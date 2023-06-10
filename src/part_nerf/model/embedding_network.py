import math
from functools import partial
from typing import Dict

import torch
from torch import nn


class SimpleEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, max_norm: float = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, max_norm=max_norm)
        # initialization
        nn.init.normal_(self.embedding.weight.data, 0.0, 1.0 / math.sqrt(embedding_dim))
        self._distribution = None

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(idx)

    def get_random_embeddings(self, num_items: int) -> torch.Tensor:
        if self._distribution is None:
            self._distribution = self.calculate_distribution(
                self.embedding.weight.clone().detach()
            )

        return self._distribution.sample((num_items,))

    @staticmethod
    def calculate_distribution(detached_weights: torch.Tensor):
        mean = detached_weights.mean(dim=0)
        detached_weights = detached_weights - mean[None, :]
        cov = torch.einsum("nd,nc->dc", detached_weights, detached_weights) / (
            detached_weights.shape[0] - 1
        )
        return torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)


def get_embedding_network(name, config: Dict):
    embedding_size = config.get("embedding_size")
    num_embeddings = config.get("num_embeddings")
    max_norm = config.get("max_norm")
    embedding_factory = {
        "simple": partial(
            SimpleEmbedding,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_size,
            max_norm=max_norm,
        ),
    }
    return embedding_factory[name]()
