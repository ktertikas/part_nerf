# Utils used in the part and shape interpolation scripts.
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import torch
from reconstruction_utils import (
    MeshGenerator,
    export_meshes_to_path,
    reconstruct_meshes_from_model,
)
from train_utils import forward_one_batch
from utils import collect_images_from_keys, load_latent_codes

from part_nerf.model import NerfAutodecoder
from part_nerf.utils import torch_container_to_numpy


def add_embedding_info(
    X: Dict[str, torch.Tensor],
    latent_path: Optional[Path],
    embedding_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    new_X = X.copy()
    if latent_path is not None:
        shape_code, texture_code = load_latent_codes(latent_path, embedding_id)
        new_X["shape_embedding"] = shape_code.to(device)
        new_X["texture_embedding"] = texture_code.to(device)
    else:
        # Loading from specified embedding id
        print(f"Loading embedding id {embedding_id} from pretrained model")
        new_X["scene_id"] = torch.tensor(
            [embedding_id], dtype=torch.long, device=device
        )
    return new_X


def run_mesh_extraction_from_part_features(
    model: NerfAutodecoder,
    mesh_generator: MeshGenerator,
    predictions: Dict[str, torch.Tensor],
    full_reconstruction_dir: Path,
    part_reconstruction_dir: Path,
    name: str,
    id: int,
    chunk_size: int,
    device: torch.device,
    num_parts: int,
    with_parts: bool = True,
):
    model_occupancy_callable = partial(
        model.forward_occupancy_field_from_part_features, pred_dict=predictions
    )
    mesh, part_meshes_list = reconstruct_meshes_from_model(
        model_occupancy_callable,
        mesh_generator,
        chunk_size,
        device,
        with_parts=with_parts,
        num_parts=num_parts,
    )
    export_meshes_to_path(
        full_reconstruction_dir,
        part_reconstruction_dir,
        id,
        mesh,
        part_meshes_list,
        name=name,
    )


def run_volumetric_rendering_from_part_features(
    model: NerfAutodecoder,
    renderer,
    X: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    images_dir: Path,
    rays_chunk: int,
    id: int,
    name: str,
):
    model_color_callable = partial(
        model.forward_color_field_from_part_features, pred_dict=predictions
    )
    color_predictions = forward_one_batch(
        model_color_callable, renderer, X, rays_chunk=rays_chunk
    )
    detached_predictions = torch_container_to_numpy(color_predictions)
    detached_targets = torch_container_to_numpy(X)
    images = collect_images_from_keys(
        predictions=detached_predictions,
        targets=detached_targets,
        keys=["rgb"],
    )
    images["rgb"][0].save((images_dir / f"{name}_{id:04}.png"))
    del color_predictions


def interpolate_vectors(
    vec1: torch.Tensor, vec2: torch.Tensor, factor: float
) -> torch.Tensor:
    assert factor >= 0.0 and factor <= 1.0
    return factor * vec1 + (1 - factor) * vec2
