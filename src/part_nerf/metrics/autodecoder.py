from typing import Tuple

import torch


def alpha_values(alphas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, M, _ = alphas.shape
    assert alphas.shape[2] == 3

    # parse min, max and mean values across batches and primitives
    min_alphas = (alphas.min(dim=1)[0]).min(dim=0)[0]  # (3,)
    max_alphas = (alphas.max(dim=1)[0]).max(dim=0)[0]  # (3,)
    mean_alphas = alphas.mean(dim=(0, 1))  # (3, )
    return min_alphas, max_alphas, mean_alphas
