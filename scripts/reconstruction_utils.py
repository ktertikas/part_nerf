from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh
from utils import colormap

from part_nerf.external.libmcubes import marching_cubes
from part_nerf.external.libmise import MISE
from part_nerf.utils import torch_container_to_numpy


def make_3d_grid(bb_min: Tuple, bb_max: Tuple, shape: Tuple) -> torch.Tensor:
    """Makes a 3D grid.

    Args:
        bb_min (Tuple): Bounding box minimum
        bb_max (Tuple): Bounding box maximum
        shape (Tuple): Output shape

    Returns:
        torch.Tensor: The 3D grid
    """
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


class MeshGenerator:
    def __init__(
        self,
        resolution: int = 64,
        mise_resolution: int = 32,
        padding: float = 0.0,
        threshold: float = 0.5,
        upsampling_steps: int = 3,
    ):
        """Class that is responsible for the generation of meshes. Note that this class only supports
        batch size of 1!

        Args:
            resolution (int, optional): The grid resolution. Defaults to 64.
            mise_resolution (int, optional): The initial resolution used for MISE. Defaults to 32.
            padding (float, optional): The padding used for the maching cubes extraction. Defaults to 0.0.
            threshold (float, optional): The implicit field threshold. Defaults to 0.5.
            upsampling_steps (int, optional): The number of upsampling steps used in MISE. Defaults to 3.
        """
        self.resolution = resolution
        self.mise_resolution = mise_resolution
        self.padding = padding
        self.threshold = threshold
        self.upsampling_steps = upsampling_steps

    def extract_mesh(self, occ_preds: np.ndarray):
        # Some short hands
        n_x, n_y, n_z = occ_preds.shape
        box_size = 1 + self.padding
        # Make sure that mesh is watertight
        occ_hat_padded = np.pad(occ_preds, 1, "constant", constant_values=-1e6)
        vertices, triangles = marching_cubes(occ_hat_padded, self.threshold)
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices = box_size * (vertices - 0.5)

        # Create mesh
        mesh = Trimesh(vertices, triangles, process=False)
        return mesh

    def evaluate_points(
        self, model, points: np.ndarray, chunk_size: int, device: torch.device
    ) -> np.ndarray:
        X = {}
        implicit_field_list = []
        split_points_list = torch.split(points, chunk_size, dim=0)
        for point_split in split_points_list:
            # only ray_points key needed for evaluation of implicit field
            X["ray_points"] = point_split[None, None, ...].repeat(1, 1, 1, 1).to(device)
            predictions = model(X)
            detached_predictions = torch_container_to_numpy(predictions)
            implicit_field_pred = detached_predictions["implicit_field"].reshape(
                len(point_split), -1
            )
            implicit_field_list.append(implicit_field_pred)
        implicit_field_pred = np.concatenate(
            implicit_field_list, axis=0
        )  # (num_points, num_parts)
        return implicit_field_pred

    def get_mise_predictions(
        self, model, chunk_size: int, device: torch.device
    ) -> np.ndarray:
        # MISE - note this is working only for a batch of size 1!
        mesh_extractor = MISE(
            self.mise_resolution, self.upsampling_steps, self.threshold
        )
        points = mesh_extractor.query()
        while points.shape[0] != 0:
            # Query points - normalize to category bounds,
            # [-1.2, 1.2] for the lego dataset, [-0.5, 0.5] for shapenet
            pointsf = (1.0 + self.padding) * (
                torch.tensor(points) / mesh_extractor.resolution - 0.5
            )
            implicit_field_pred = self.evaluate_points(
                model, pointsf, chunk_size, device
            )
            values = implicit_field_pred.max(axis=-1).astype(
                np.float64
            )  # (point_split_len,)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()
        implicit_field_pred = mesh_extractor.to_dense()[..., None]
        return implicit_field_pred

    def get_mise_predictions_per_part(
        self, model, chunk_size: int, device: torch.device, num_parts: int
    ):
        per_part_field = []
        for i in range(num_parts):
            part_mesh_extractor = MISE(
                self.mise_resolution, self.upsampling_steps, self.threshold
            )
            points = part_mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points - normalize to category bounds,
                # [-1.2, 1.2] for the lego dataset, [-0.5, 0.5] for shapenet
                pointsf = (1.0 + self.padding) * (
                    torch.tensor(points) / part_mesh_extractor.resolution - 0.5
                )
                implicit_field_pred = self.evaluate_points(
                    model, pointsf, chunk_size, device
                )
                part_values = implicit_field_pred[..., i].astype(
                    np.float64
                )  # (point_split_len,)
                part_mesh_extractor.update(points, part_values)
                points = part_mesh_extractor.query()
            part_field_pred = part_mesh_extractor.to_dense()[..., None]
            per_part_field.append(part_field_pred)
        implicit_field_pred = np.concatenate(per_part_field, axis=-1)
        return implicit_field_pred

    def get_standard_predictions(
        self, model, chunk_size: int, device: torch.device
    ) -> np.ndarray:
        scale = 1.0 + self.padding
        grid = make_3d_grid(
            (-0.5 * scale,) * 3, (0.5 * scale,) * 3, (self.resolution,) * 3
        )
        implicit_field_pred = self.evaluate_points(model, grid, chunk_size, device)
        return implicit_field_pred.reshape(
            (self.resolution, self.resolution, self.resolution, -1)
        )


def reconstruct_meshes_from_model(
    model_callable,
    mesh_generator: MeshGenerator,
    chunk_size: int,
    device: torch.device,
    with_parts: bool = False,
    num_parts: int = None,
):
    # Run MISE for full mesh reconstruction
    implicit_field_pred = mesh_generator.get_mise_predictions(
        model_callable, chunk_size, device
    )
    max_occupancy = np.max(implicit_field_pred, axis=-1)
    mesh = mesh_generator.extract_mesh(max_occupancy)

    part_meshes_list = []
    if with_parts:
        assert (
            num_parts is not None
        ), "Need to define number of parts for reconstruction"
        # Standard reconstruction for parts predictions
        implicit_field_parts_pred = mesh_generator.get_mise_predictions_per_part(
            model_callable, chunk_size, device, num_parts
        )
        # Per primitive reconstruction
        part_colors = colormap(np.linspace(0, 1, num_parts))
        for i in range(num_parts):
            part_mesh = mesh_generator.extract_mesh(implicit_field_parts_pred[..., i])
            part_color_uint8 = np.uint8(part_colors[i] * 255)
            part_mesh.visual.vertex_colors = part_color_uint8
            part_meshes_list.append(part_mesh)
    return mesh, part_meshes_list


def export_meshes_to_path(
    full_reconstruction_dir: Path,
    part_reconstruction_dir: Path,
    embedding_id: int,
    mesh: Trimesh,
    part_meshes_list: List[Trimesh],
    name: str = "original",
):
    mesh_path = full_reconstruction_dir / f"{name}_{embedding_id:04}.obj"
    export_mesh(mesh, mesh_path)
    for j, part_mesh in enumerate(part_meshes_list):
        export_mesh(
            part_mesh,
            (part_reconstruction_dir / f"{name}_{embedding_id:04}_{j:02}.obj"),
        )
