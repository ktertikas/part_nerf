import torch
from torch import nn

from ..primitive_utils import ellipsoid_volume


def ray_max_point_crossentropy(
    implicit_field: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    B, N, _, _ = implicit_field.shape
    # inside_occupancy_points_mask = implicit_field > 0.5  # (B, N, num_points, M)
    # inside_pred_field = implicit_field * inside_occupancy_points_mask  # (B, N, num_points, M)
    # first inside point across ray
    max_ray_point_val = torch.max(implicit_field, dim=2)[0]  # (B, N, M)
    # max ray occupancy value across primitives
    max_ray_val = torch.max(max_ray_point_val, dim=-1)[0]  # (B, N)

    assert labels.shape == (B, N, 1)

    return nn.functional.binary_cross_entropy(max_ray_val, labels[..., 0])


def ray_coverage_loss(
    implicit_field: torch.Tensor, labels: torch.Tensor, num_inside_rays: int = 10
) -> torch.Tensor:
    B, N, _, _ = implicit_field.shape
    assert labels.shape == (B, N, 1)

    # first inside point across ray
    max_ray_point_val = torch.max(implicit_field, dim=2)[0]  # (B, N, M)
    # mask rays with negative label
    positive_rays_implicit = max_ray_point_val * labels
    # get topk rays for each primitive
    top_k_rays_implicit = torch.topk(
        positive_rays_implicit, num_inside_rays, dim=1, largest=True, sorted=False
    )[0]
    # Compute the loss for the topk implicit rays
    sm = implicit_field.new_tensor(1e-6)
    loss = -torch.log(top_k_rays_implicit + sm)
    return loss.mean()


def ray_overlapping_loss(
    implicit_field: torch.Tensor, labels: torch.Tensor, max_hitting_primitives: int = 3
) -> torch.Tensor:
    B, N, _, _ = implicit_field.shape
    assert labels.shape == (B, N, 1)

    # first inside point across ray
    max_ray_point_val = torch.max(implicit_field, dim=2)[0]  # (B, N, M)

    # sum across primitives
    max_ray_primitive_sum = torch.sum(max_ray_point_val, dim=-1)  # (B, N)
    zeros_tensor = torch.zeros_like(max_ray_primitive_sum)

    overlapping_term = torch.maximum(
        max_ray_primitive_sum - max_hitting_primitives, zeros_tensor
    )
    return overlapping_term.mean()


def ellipsoid_volume_consistency_loss(alphas: torch.Tensor) -> torch.Tensor:
    B, M, _ = alphas.shape
    volumes = ellipsoid_volume(alphas)  # (B, M)
    loss = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            loss += torch.abs(volumes[:, i] - volumes[:, j]).sum()
    num_items = B * M * (M - 1) / 2.0
    if num_items > 0.0:
        loss /= num_items
    return loss
