from functools import partial
from typing import Dict

import torch
from torch import nn


class RayPointsOccupancyAssociator(nn.Module):
    """Implementation that associates each ray to the closest primitive, where each
    primitive is defined by the per-primitive conditioned occupancy decoder. In particular,
    we use the implicit values for each point sampled along each ray and we associate a ray
    by the primitive that hass the first inside point (>= implicit threshold) along the ray.
    This should behave better than the argmax along the ray."""

    def __init__(self, implicit_threshold: float = 0.5):
        super().__init__()
        self._implicit_threshold = implicit_threshold

    def forward(
        self, X: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        occupancy_values = pred_dict["implicit_field"]  # (B, N, num_points, M)

        # Max implicit field value for every point and associated index
        max_occupancy_per_point, primitive_indices_per_point = occupancy_values.max(
            dim=-1
        )  # (B, N, num_points), (B, N, num_points)

        # Mask denoting if points are inside or outside from the implicit field
        inside_occupancy_per_ray = max_occupancy_per_point >= self._implicit_threshold
        # Get point indices for the first True value, and inside-outside mask per ray
        positive_rays, positive_point_indices = inside_occupancy_per_ray.max(
            dim=-1
        )  # (B, N), (B, N)

        # Primitive indices
        primitive_indices = torch.gather(
            primitive_indices_per_point, dim=-1, index=positive_point_indices[..., None]
        )[
            ..., 0
        ]  # (B, N)

        pred_dict["no_rendering"] = ~positive_rays  # (B, N)
        pred_dict["primitive_associations"] = primitive_indices
        return pred_dict


def get_ray_associator(cfg: Dict) -> nn.Module:
    name = cfg["type"]
    ray_associator_factory = {
        "occupancy": partial(
            RayPointsOccupancyAssociator,
            implicit_threshold=cfg.get("implicit_threshold", 0.5),
        ),
    }
    return ray_associator_factory[name]()
