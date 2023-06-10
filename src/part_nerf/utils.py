import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def torch_container_to_numpy(x):
    if isinstance(x, dict):
        return {k: torch_container_to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)([torch_container_to_numpy(i) for i in x])
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").numpy()
    if isinstance(x, (np.ndarray, float, int)):
        return x
    raise ValueError(
        f"{type(x)} is not currently supported for converting to detached numpy container"
    )


def send_to(x: Any, device: torch.device):
    if isinstance(x, dict):
        return {k: send_to(v, device=device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)([send_to(i, device=device) for i in x])
    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    return x


def dict_to_device_and_batchify(X: Dict, device: torch.device) -> Dict:
    # Send to device and add batch dimension
    for k, v in X.items():
        if isinstance(v, np.ndarray):
            X[k] = torch.from_numpy(v)
        if isinstance(v, (int, float)):
            X[k] = torch.tensor(v)
        X[k] = X[k][None].to(device=device)
    return X


def farthest_point_sampling(
    points: torch.Tensor, num_samples: int, return_index: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    device = points.device
    B, N, C = points.shape

    sampled = torch.zeros((B, num_samples, C), dtype=points.dtype, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    indices = torch.zeros((B, num_samples), dtype=torch.long, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(num_samples):
        indices[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, C)
        sampled[:, i, :] = centroid[:, 0, :]
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    if return_index:
        return sampled, indices
    return sampled


def shifted_cumprod(x, shift: int = 1) -> torch.Tensor:
    """
    Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of
    ones and removes `shift` trailing elements to/from the last dimension
    of the result.

    Code adapted from
    https://github.com/facebookresearch/pytorch3d/blob/1701b76a31e3e8c97d51b49dfcaa060762ab3764/pytorch3d/renderer/implicit/raymarching.py#L165
    """
    x_cumprod = torch.cumprod(x, dim=-1)
    x_cumprod_shift = torch.cat(
        [torch.ones_like(x_cumprod[..., :shift]), x_cumprod[..., :-shift]], dim=-1
    )

    return x_cumprod_shift


def save_checkpoints(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    experiment_directory: Path,
    scheduler: Optional[_LRScheduler] = None,
) -> None:
    print(f"Saving model and optimizer checkpoints in {experiment_directory}")
    torch.save(model.state_dict(), experiment_directory / f"model_{epoch:05d}")
    torch.save(optimizer.state_dict(), experiment_directory / f"opt_{epoch:05d}")
    if scheduler is not None:
        torch.save(
            scheduler.state_dict(), experiment_directory / f"scheduler_{epoch:05d}"
        )


def load_checkpoints(
    model: nn.Module,
    optimizer: Optimizer,
    experiment_directory: Path,
    config: Dict,
    device: torch.device,
    scheduler: Optional[_LRScheduler] = None,
    model_id: int = None,
):
    model_files = [
        f.name for f in experiment_directory.iterdir() if f.name.startswith("model_")
    ]
    if len(model_files) == 0:
        print("Empty experiment directory, no checkpoint loading")
        return
    ids = [int(f[6:]) for f in model_files]
    if model_id is not None:
        if model_id in ids:
            selected_id = model_id
        else:
            raise FileNotFoundError(
                f"Model id {model_id} was not found in {experiment_directory}"
            )
    else:
        selected_id = max(ids)
    model_path = experiment_directory / f"model_{selected_id:05d}"
    opt_path = experiment_directory / f"opt_{selected_id:05d}"
    scheduler_path = experiment_directory / f"scheduler_{selected_id:05d}"
    if model_path.exists():
        print(f"Loading model checkpoint from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    if opt_path.exists() and optimizer is not None:
        print(f"Loading optimizer checkpoint from {opt_path}")
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    if scheduler_path.exists() and scheduler is not None:
        print(f"Loading scheduler checkpoint from {scheduler_path}")
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
    if config.get("trainer") is not None:
        config["trainer"]["start_epoch"] = selected_id + 1


def split_rays_dict(
    ds: Dict[str, torch.Tensor],
    rays_chunk: int,
    split_keys: List[str] = [
        "ray_origins",
        "ray_directions",
        "ray_points",
        "ray_lengths",
        "sampled_rows",
        "sampled_cols",
        "colors",
    ],
    dim: int = 1,
) -> List[Dict[str, torch.Tensor]]:
    # assuming rays are in the 2nd dimension of the torch Tensor, right after
    # the batch dimension
    assert len(split_keys) > 0
    num_splits = math.ceil(ds[split_keys[0]].shape[dim] / rays_chunk)
    list_of_dicts = [dict() for _ in range(num_splits)]
    for k, v in ds.items():
        if k in split_keys:
            v_lists = torch.split(v, rays_chunk, dim=dim)
            for i, v_split in enumerate(v_lists):
                # adding key to respective dictionary
                list_of_dicts[i][k] = v_split
        else:
            for i in range(num_splits):
                list_of_dicts[i][k] = v

    return list_of_dicts


def merge_rays_predictions(
    pred_list: List[Dict[str, torch.Tensor]],
    merge_keys: List[str] = [
        "ray_densities",
        "ray_colors",
        "inside_outside",
        "implicit_field",
        "coarse_implicit_field",
        "primitive_associations",
        "no_rendering",
        "normals",
    ],
    dim: int = 1,
) -> Dict[str, torch.Tensor]:
    # assuming rays are in the 2nd dimension of the torch Tensor, right after
    # the batch dimension
    assert len(pred_list) > 0
    pred_keys = list(pred_list[0])
    merged_dict = {}
    for k in pred_keys:
        if k in merge_keys:
            merged_dict[k] = torch.cat([ds[k] for ds in pred_list], dim=dim)
        else:
            # take pred value from 1st dataset in the list
            merged_dict[k] = pred_list[0][k]
    return merged_dict


def batchify_rays(
    fn,
    rays_chunk: int = None,
    split_keys: List[str] = [
        "ray_origins",
        "ray_directions",
        "ray_points",
        "ray_lengths",
        "sampled_rows",
        "sampled_cols",
        "colors",
    ],
    merge_keys: List[str] = [
        "ray_densities",
        "ray_colors",
        "inside_outside",
        "implicit_field",
        "coarse_implicit_field",
        "primitive_associations",
        "no_rendering",
    ],
    dim: int = 1,
) -> Dict[str, torch.Tensor]:
    if rays_chunk is None:
        return fn

    def ret(input_dict):
        return merge_rays_predictions(
            [
                fn(X)
                for X in split_rays_dict(
                    input_dict, rays_chunk, split_keys=split_keys, dim=dim
                )
            ],
            merge_keys=merge_keys,
            dim=dim,
        )

    return ret
