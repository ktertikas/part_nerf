from functools import partial
from typing import Dict

import torch
from torch import nn

from .utils import shifted_cumprod


class SingleOccupancyRayMarcher(nn.Module):
    """Class that implements an Occupancy Ray Marcher that assumes only 1 occupancy value and
    color value, and renders using the ideas from the UNISURF paper.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.white_bkgd = config.get("white_background", False)

    def forward(
        self,
        ray_lengths: torch.Tensor,
        ray_directions: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """From the predicted occupancies per primitive and RGB colors render various images
           such as an RGB image containing the scene, a depth map, a disparity
           map etc.

        Args:
            ray_lengths (torch.Tensor): A torch.tensor of size BxNxM, containing the depth
                values of 3D points along the ray.
            predictions (Dict[str, torch.Tensor]): A dictionary containing the predictions
                from the NeRF module.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the predicted values.
        """
        # Extract the batch size, number of rays and number of points along the
        # ray
        B, N, M = ray_lengths.shape

        ray_colors = predictions["ray_colors"]  # (B, N, M, 3)
        ray_occupancies_per_primitive = predictions[
            "implicit_field"
        ]  # (B, N, M, num_parts)
        no_rendering_rays = predictions["no_rendering"]  # (B, N)
        ray_occupancies = torch.max(ray_occupancies_per_primitive, dim=-1)[
            0
        ]  # (B, N, M)
        assert ray_colors.shape == (B, N, M, 3)
        assert ray_occupancies.shape == (B, N, M)

        # Make sure that ray_occupancies are between 0 and 1
        # self._check_occupancy_bounds(ray_densities)

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        weights = ray_occupancies * shifted_cumprod(1 - ray_occupancies + 1e-10)
        assert weights.shape == (B, N, M)

        # No rendering rays get zeroed weight values
        weights = weights * (~no_rendering_rays[..., None])

        # Compute the weighted color of each sample along each ray.
        rgb_map = torch.sum(weights[..., None] * ray_colors, dim=-2)
        assert rgb_map.shape == (B, N, 3)  # A color per ray

        # Estimated depth map is expected distance.
        depth_map = torch.sum(weights * ray_lengths, dim=-1)

        # Disparity map is inverse depth.
        dmp = torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
        )
        disp_map = 1.0 / dmp

        # Sum of weights along each ray. This value is in [0, 1] up to
        # numerical error.
        acc_map = torch.sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if self.white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return {
            "rgb": rgb_map,
            "depth": depth_map[..., None],
            "disparity": disp_map[..., None],
            "masks": acc_map[..., None],
            "weights": weights,
        }


def build_renderer(config: Dict) -> SingleOccupancyRayMarcher:
    type = config.get("type", "occ_single_nerf")
    renderer_factory = {
        "occ_single_nerf": partial(SingleOccupancyRayMarcher, config=config),
    }
    return renderer_factory[type]()
