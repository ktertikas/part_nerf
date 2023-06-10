from typing import Dict

import torch

from .mse_loss import mse_loss, mse_loss_positive
from .rays_losses import (
    ellipsoid_volume_consistency_loss,
    ray_coverage_loss,
    ray_max_point_crossentropy,
    ray_overlapping_loss,
)


def calculate_loss(
    name: str,
    input: torch.Tensor = None,
    predictions: Dict[str, torch.Tensor] = None,
    targets: Dict[str, torch.Tensor] = None,
    extra_args: Dict = {},
) -> Dict[str, torch.Tensor]:
    if name == "mse_loss_coarse":
        loss = mse_loss(
            predictions=predictions["rgb_coarse"], targets=targets["colors"]
        )
    elif name == "mse_loss_positive_coarse":
        loss = mse_loss_positive(
            predictions=predictions["rgb_coarse"],
            targets=targets["colors"],
            no_rendering_rays_mask=~(targets["gt_mask"][..., 0] > 0)
            * predictions["no_rendering"],
        )
    elif name == "mse_loss":
        loss = mse_loss(predictions=predictions["rgb"], targets=targets["colors"])
    elif name == "mse_loss_positive":
        loss = mse_loss_positive(
            predictions=predictions["rgb"],
            targets=targets["colors"],
            no_rendering_rays_mask=~(targets["gt_mask"][..., 0] > 0)
            * predictions["no_rendering"],
        )
    elif name == "ray_max_point_crossentropy":
        loss = ray_max_point_crossentropy(
            implicit_field=predictions["implicit_field"],
            labels=targets["gt_mask"],
        )
    elif name == "ray_max_point_crossentropy_coarse":
        loss = ray_max_point_crossentropy(
            implicit_field=predictions["coarse_implicit_field"],
            labels=targets["gt_mask"],
        )
    elif name == "ray_coverage_loss":
        loss = ray_coverage_loss(
            implicit_field=predictions["implicit_field"],
            labels=targets["gt_mask"],
            num_inside_rays=extra_args.get("num_inside_rays", 10),
        )
    elif name == "ray_overlapping_loss_coarse":
        loss = ray_overlapping_loss(
            implicit_field=predictions["coarse_implicit_field"],
            labels=targets["gt_mask"],
            max_hitting_primitives=extra_args.get("max_hitting_primitives", 3),
        )
    elif name == "mask_loss_positive":
        loss = mse_loss_positive(
            predictions=predictions["masks"],
            targets=targets["gt_mask"],
            no_rendering_rays_mask=predictions["no_rendering"],
        )
    elif name == "mask_loss":
        loss = mse_loss(predictions=predictions["masks"], targets=targets["gt_mask"])
    elif name == "mask_loss_positive_coarse":
        loss = mse_loss_positive(
            predictions=predictions["masks_coarse"],
            targets=targets["gt_mask"],
            no_rendering_rays_mask=predictions["no_rendering"],
        )
    elif name == "mask_loss_coarse":
        loss = mse_loss(
            predictions=predictions["masks_coarse"], targets=targets["gt_mask"]
        )
    elif name == "shape_embedding_normalization_loss":
        loss = torch.norm(predictions["shape_embedding"], dim=-1).mean()
    elif name == "texture_embedding_normalization_loss":
        loss = torch.norm(predictions["texture_embedding"], dim=-1).mean()
    elif name == "volume_consistency_loss":
        loss = ellipsoid_volume_consistency_loss(alphas=predictions["scale"])
    else:
        raise NotImplementedError(f"Loss with name {name} has not been implemented.")
    return {name: loss}


def calculate_losses(
    loss_config: Dict,
    input: torch.tensor = None,
    predictions: Dict[str, torch.tensor] = None,
    targets: Dict[str, torch.tensor] = None,
) -> Dict[str, torch.Tensor]:
    loss_names_list = loss_config.get("type")
    loss_weights_list = loss_config.get("weights")
    if len(loss_weights_list) == 0:
        loss_weights_list = [1.0] * len(loss_names_list)
    total_loss = 0.0
    loss_dict = {}
    for i, loss_name in enumerate(loss_names_list):
        intermediate_loss_dict = calculate_loss(
            loss_name,
            input=input,
            predictions=predictions,
            targets=targets,
            extra_args=loss_config,
        )
        loss_dict.update(intermediate_loss_dict)
        total_loss += loss_weights_list[i] * intermediate_loss_dict[loss_name]
    loss_dict["total_loss"] = total_loss
    return loss_dict
