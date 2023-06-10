from typing import Dict, Tuple

import torch
from torch import nn

from part_nerf.model.utils import sample_pdf
from part_nerf.utils import batchify_rays


def forward_one_batch(
    model: nn.Module,
    renderer: nn.Module,
    X: Dict,
    rays_chunk: int = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    ray_lengths = X["ray_lengths"]
    ray_directions = X["ray_directions"]
    # Forward pass
    predictions = batchify_rays(model, rays_chunk=rays_chunk)(X)
    renders = renderer(ray_lengths, ray_directions, predictions)
    predictions.update(renders)
    return predictions


def forward_one_batch_coarse_fine(
    model: nn.Module,
    renderer: nn.Module,
    X: Dict,
    rays_chunk: int = None,
    num_samples: int = 32,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    ray_lengths = X["ray_lengths"]
    ray_directions = X["ray_directions"]
    ray_origins = X["ray_origins"]
    # Forward pass - coarse points
    predictions_coarse = batchify_rays(model, rays_chunk=rays_chunk)(X)
    renders = renderer(ray_lengths, ray_directions, predictions_coarse)
    predictions_coarse.update(renders)

    # Extract the weights to perform the inverse transform sampling as
    # described in Section 3.3. in the original paper.
    weights = predictions_coarse["weights"]

    # Obtain additional integration times to evaluate based on the weights
    # assigned to colors in the coarse model
    # Code adapted from
    # https://github.com/yenchenlin/nerf-pytorch/blob/a15fd7cb363e93f933012fd1f1ad5395302f63a4/run_nerf.py#L392
    # and from the original tf implementation
    # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L213
    z_vals = ray_lengths
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(mids, weights[..., 1:-1], num_samples, uniform=False)
    z_samples = z_samples.detach()

    # Obtain all points to evaluate color and density at. Note that we also
    # evaluate *again* to the points that we already used for predicting
    # colours and opacities using the coarse network.
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    ray_points = (
        ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
    )
    X["ray_points"] = ray_points
    X["ray_lengths"] = z_vals
    ray_lengths = z_vals

    # Forward pass: fine + coarse points
    predictions = batchify_rays(model, rays_chunk=rays_chunk)(X)
    renders = renderer(ray_lengths, ray_directions, predictions)
    predictions.update(renders)

    # append predictions of coarse network
    for k, v in predictions_coarse.items():
        predictions[f"{k}_coarse"] = v

    return predictions
