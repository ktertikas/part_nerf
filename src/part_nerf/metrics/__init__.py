from typing import Dict

import torch

from .autodecoder import alpha_values
from .psnr import psnr_metric


def calculate_metric(
    name: str,
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    extra_args: Dict = {},
) -> Dict[str, torch.Tensor]:
    if name == "psnr":
        psnr = psnr_metric(
            pred_rgb=predictions["rgb"],
            gt_rgb=targets["colors"],
        )
        metric_dict = {"psnr": psnr}
    elif name == "psnr_coarse":
        psnr = psnr_metric(
            pred_rgb=predictions["rgb_coarse"],
            gt_rgb=targets["colors"],
        )
        metric_dict = {"psnr": psnr}
    elif name == "scale":
        min_scale, max_scale, mean_scale = alpha_values(predictions["scale"])
        metric_dict = {
            "scale_0_min": min_scale[0].item(),
            "scale_1_min": min_scale[1].item(),
            "scale_2_min": min_scale[2].item(),
            "scale_0_max": max_scale[0].item(),
            "scale_1_max": max_scale[1].item(),
            "scale_2_max": max_scale[2].item(),
            "scale_0_mean": mean_scale[0].item(),
            "scale_1_mean": mean_scale[1].item(),
            "scale_2_mean": mean_scale[2].item(),
        }
    elif name == "associator":
        outside_rays = predictions["no_rendering"]
        metric_dict = {
            "inside_rays": (~outside_rays * 1.0).sum(dim=-1).mean().item(),
            "outside_rays": (outside_rays * 1.0).sum(dim=-1).mean().item(),
        }
    else:
        raise NotImplementedError(f"Metric with name {name} has not been implemented.")
    return metric_dict


def calculate_metrics(
    metric_config: Dict,
    predictions: Dict[str, torch.tensor] = None,
    targets: Dict[str, torch.tensor] = None,
) -> Dict[str, torch.Tensor]:
    metric_names_list = metric_config.get("type")
    metric_dict = {}
    for m_name in metric_names_list:
        metric_dict.update(
            calculate_metric(
                m_name,
                predictions=predictions,
                targets=targets,
                extra_args=metric_config,
            )
        )
    return metric_dict
