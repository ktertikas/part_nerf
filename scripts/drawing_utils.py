from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import trimesh
from simple_3dviz import Mesh, Scene, Spherecloud, render
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.renderables import Renderable
from trimesh.visual.color import to_float
from utils import colormap, id_generator, numpy_images_to_pil_batched
from wandb import Image, Video

from part_nerf.primitive_utils import quaternions_to_rotation_matrices_np
from part_nerf.stats_logger import WandB

CAMERA_POSITION = {
    "shapenet": (0, 0, 1.8),
    "lego_dataset": (0, -4, 0),
}

UP_VECTOR = {
    "shapenet": (0, 1, 0),
    "lego_dataset": (0, 0, 1),
}

# ModernGL context messes up things, so defining 1 scene for the entire module
GENERIC_DRAWING_SCENE = Scene(size=(512, 512))


def parse_points_to_sphereclouds(
    points: np.ndarray,
    colors: Union[Tuple[float], List[float]] = (0.3, 0.3, 0.3, 1.0),
    sizes: float = 0.02,
) -> Spherecloud:
    pts = Spherecloud(points, colors=colors, sizes=sizes)
    return pts


def image_from_renderable_list(
    renderable_list: List[Renderable],
    save_path: str,
    camera_position: Union[List[float], Tuple[float]] = (0, -1.8, 0),
    up_vector: Union[List[int], Tuple[int]] = (0, 0, 1),
    background: Union[List[int], Tuple[int]] = (1.0, 1.0, 1.0, 1.0),
    rendering_scene: Scene = None,
):
    behavior = SaveFrames(save_path)
    render(
        renderable_list,
        behaviours=[behavior],
        n_frames=1,
        background=background,
        camera_position=camera_position,
        up_vector=up_vector,
        scene=rendering_scene,
    )


def gif_from_renderable_list(
    renderable_list: List[Renderable],
    save_path: str,
    camera_position: Union[List[float], Tuple[float]] = (0, -1.8, 0),
    up_vector: Union[List[int], Tuple[int]] = (0, 0, 1),
    background: Union[List[int], Tuple[int]] = (1.0, 1.0, 1.0, 1.0),
    rendering_scene: Scene = None,
):
    behavior_list = [
        LightToCamera(),
        CameraTrajectory(Circle((0, 0, 0), camera_position, up_vector), speed=1 / 180),
        SaveGif(save_path),
    ]
    render(
        renderable_list,
        behaviours=behavior_list,
        n_frames=180,
        background=background,
        camera_position=camera_position,
        up_vector=up_vector,
        scene=rendering_scene,
    )


def add_reconstruction_data_to_logger(
    predictions: Dict[str, np.ndarray],
    logger: WandB,
    batch_idx: int,
    full_reconstruction_dir: Path,
    part_reconstruction_dir: Path,
    data_type: str = "deforming_things_4d",
    epoch: int = -1,
) -> None:
    # camera position and up vector based on data type
    camera_pos = CAMERA_POSITION[data_type]
    up_vector = UP_VECTOR[data_type]
    # parsing occupancy field
    occupancy_field_grid = predictions.get("implicit_field")
    B = occupancy_field_grid.shape[0]
    # Entire object mesh reconstruction
    assert len(occupancy_field_grid.shape) == 5
    # setting up colors for part based reconstruction
    M = occupancy_field_grid.shape[-1]
    # Use the existing predefined rendering scene
    rendering_scene = GENERIC_DRAWING_SCENE
    for i in range(B):
        # Entire object mesh rendering
        mesh_path = full_reconstruction_dir / f"mesh_{batch_idx:04}_{i:04}.obj"
        reconstructed_mesh_renderable = Mesh.from_file(
            str(mesh_path), color=(0.5, 0.5, 0.5)
        )

        # Plot per item in the batch
        tmp_path = f"/tmp/mesh_img_reconstruction_{id_generator()}.png"
        image_from_renderable_list(
            [reconstructed_mesh_renderable],
            tmp_path,
            rendering_scene=rendering_scene,
            camera_position=camera_pos,
            up_vector=up_vector,
        )
        logger.add_media(
            epoch,
            Image(tmp_path),
            f"mesh_reconstruction_image_{batch_idx:04}_{i:04}",
        )

        # Gif animations
        tmp_path = f"/tmp/mesh_gif_reconstruction_{id_generator()}.gif"
        gif_from_renderable_list(
            [reconstructed_mesh_renderable],
            tmp_path,
            rendering_scene=rendering_scene,
            camera_position=camera_pos,
            up_vector=up_vector,
        )
        logger.add_media(
            epoch, Video(tmp_path), f"mesh_reconstruction_gif_{batch_idx:04}_{i:04}"
        )

        # Parts mesh reconstruction rendering
        part_meshes_renderables = []
        for j in range(M):
            mesh_part_path = (
                part_reconstruction_dir / f"part_mesh_{batch_idx:04}_{i:04}_{j:02}.obj"
            )
            try:
                m = Mesh.from_file(str(mesh_part_path))
            except IndexError:
                # happens when part mesh is empty
                print(f"{mesh_part_path} is an empty mesh, skipping!")
                continue
            # loading using trimesh to get proper colors
            m2 = trimesh.load(mesh_part_path, process=False, force="mesh")
            m.colors = to_float(
                m2.visual.vertex_colors[0, :3]
            )  # only take first vertex color
            part_meshes_renderables.append(m)

        # Plot per item in the batch
        tmp_path = f"/tmp/part_mesh_img_reconstruction_{id_generator()}.png"
        image_from_renderable_list(
            part_meshes_renderables,
            tmp_path,
            rendering_scene=rendering_scene,
            camera_position=camera_pos,
            up_vector=up_vector,
        )
        logger.add_media(
            epoch,
            Image(tmp_path),
            f"part_mesh_reconstruction_image_{batch_idx:04}_{i:04}",
        )

        # Gif animations
        tmp_path = f"/tmp/part_mesh_gif_reconstruction_{id_generator()}.gif"
        gif_from_renderable_list(
            part_meshes_renderables,
            tmp_path,
            rendering_scene=rendering_scene,
            camera_position=camera_pos,
            up_vector=up_vector,
        )
        logger.add_media(
            epoch,
            Video(tmp_path),
            f"part_mesh_reconstruction_gif_{batch_idx:04}_{i:04}",
        )


def add_nerf_reconstruction_data_to_logger(
    logger: WandB,
    data_type: str = "lego_dataset",
    epoch: int = -1,
) -> None:
    experiment_dir = Path(logger.experiment_dir)
    reconstruction_dir = experiment_dir / "reconstructions"

    camera_pos = CAMERA_POSITION[data_type]
    up_vector = UP_VECTOR[data_type]
    # Use the existing predefined rendering scene
    rendering_scene = GENERIC_DRAWING_SCENE
    # Assuming only full reconstructions here
    for mesh_path in reconstruction_dir.glob("*.obj"):
        reconstructed_mesh_renderable = Mesh.from_file(str(mesh_path))

        tmp_path = f"/tmp/{mesh_path.stem}.png"
        image_from_renderable_list(
            [reconstructed_mesh_renderable],
            tmp_path,
            rendering_scene=rendering_scene,
            camera_position=camera_pos,
            up_vector=up_vector,
        )
        logger.add_media(
            epoch,
            Image(tmp_path),
            f"{mesh_path.stem}_img",
        )

        # Gif animations
        tmp_path = f"/tmp/{mesh_path.stem}.gif"
        gif_from_renderable_list(
            [reconstructed_mesh_renderable],
            tmp_path,
            rendering_scene=rendering_scene,
            camera_position=camera_pos,
            up_vector=up_vector,
        )
        logger.add_media(epoch, Video(tmp_path), f"{mesh_path.stem}_gif")


def add_nerf_primitive_data_to_logger(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    logger: WandB,
    batch_idx: int,
    data_type: str = "lego_dataset",
    epoch: int = -1,
) -> None:
    # up vector and rotation axis based on data type
    camera_pos = CAMERA_POSITION[data_type]
    up_vector = UP_VECTOR[data_type]

    ray_points = targets.get("ray_points")  # (B, N, num_points, 3)
    implicit_field = predictions.get("implicit_field")  # (B, N, num_points, M)

    # data for per-primitive visualizations
    pred_rgb = predictions.get("rgb")  # (B, N, 3)
    ray_associations = predictions.get("primitive_associations")  # (B, N)
    no_rendering_rays = predictions.get("no_rendering")  # (B, N)
    H, W = targets["H"], targets["W"]  # (B,), (B,)
    rows_idx, cols_idx = (
        targets["sampled_rows"],
        targets["sampled_cols"],
    )  # (B, N), (B, N)

    # parsing ellipsoid predictions
    translations = predictions.get("translations", None)
    B, M, _ = translations.shape
    colors = colormap(np.linspace(0, 1, M))
    primitive_colors_with_transparency = np.hstack([colors, 0.5 * np.ones((M, 1))])
    rotations = predictions.get("rotations", None)
    if rotations is not None:
        rotations = quaternions_to_rotation_matrices_np(
            rotations.reshape((-1, 4))
        ).reshape((B, M, 3, 3))

    alphas = predictions.get("scale", None)
    # Use the existing predefined rendering scene
    rendering_scene = GENERIC_DRAWING_SCENE

    # Flag to draw ellipsoids if available
    draw_ellipsoids = (
        (translations is not None) and (rotations is not None) and (alphas is not None)
    )
    if draw_ellipsoids:
        epsilons = np.ones((B, M, 2), dtype=np.float32)
        # Plot per item in the batch
        for i in range(B):
            sq_renderables = Mesh.from_superquadrics(
                alpha=alphas[i],
                epsilon=epsilons[i],
                translation=translations[i],
                rotation=rotations[i],
                colors=primitive_colors_with_transparency,
            )
            tmp_sq_path = f"/tmp/rendered_ellipsoid_img_{id_generator(9)}.png"
            image_from_renderable_list(
                [sq_renderables],
                tmp_sq_path,
                rendering_scene=rendering_scene,
                camera_position=camera_pos,
                up_vector=up_vector,
                background=(0.0, 0.0, 0.0, 1),
            )
            logger.add_media(
                epoch,
                Image(tmp_sq_path),
                f"ellipsoid_image_{batch_idx:04}_{i:03}",
            )

            if i == 0:
                tmp_sq_path = f"/tmp/rendered_ellipsoid_gif_{id_generator(9)}.gif"
                gif_from_renderable_list(
                    [sq_renderables],
                    tmp_sq_path,
                    rendering_scene=rendering_scene,
                    camera_position=camera_pos,
                    up_vector=up_vector,
                    background=(0.0, 0.0, 0.0, 1),
                )
                logger.add_media(
                    epoch, Video(tmp_sq_path), f"ellipsoid_video_{batch_idx:04}_000"
                )
    # Flag to draw implicit field if available
    draw_implicit_field = implicit_field is not None
    if draw_implicit_field:
        # only associated ray points will be rendered, with threshold values over 0.5
        # make objects smaller for rendering purposes
        implicit_field = implicit_field.astype(np.float16)
        ray_points = ray_points.astype(np.float16)
        B, N, num_points, M = implicit_field.shape
        colors = colormap(np.linspace(0, 1, M))
        assert ray_points.shape == (B, N, num_points, 3)

        for i in range(B):
            batch_points = ray_points[i]  # (N, num_points, 3)
            batch_field = implicit_field[i]  # (N, num_points, M)
            if no_rendering_rays is not None:
                batch_rendering = ~no_rendering_rays[i]  # (N)
            else:
                batch_rendering = np.ones(N).astype(np.bool)

            # For a different visualization scheme, uncomment
            # batch_ray_associations = ray_associations[i] # (N)
            # implicit_values_assigned = batch_field[np.arange(N),:, batch_ray_associations] # (N, num_points)
            # batch_point_colors = np.tile(colors[batch_ray_associations][:, None, :], (1, num_points, 1))# (N, num_points,3)
            implicit_values_assigned = np.max(batch_field, axis=-1)  # (N, num_points)
            closest_primitive = np.argmax(batch_field, axis=-1)
            batch_point_colors = colors[closest_primitive]

            # mask for ray points that are predicted to be inside and belong in rendered rays
            inside_points_mask = (implicit_values_assigned >= 0.5) * (
                batch_rendering[..., None]
            )  # (N, num_points)
            num_inside_points = inside_points_mask.sum()
            if num_inside_points > 0:
                flattened_mask = inside_points_mask.reshape((N * num_points))
                # there exist points to be rendered
                rendering_points = batch_points.reshape((N * num_points, -1))[
                    flattened_mask
                ]  # (inside_points, 3)
                rendering_colors = batch_point_colors.reshape((N * num_points, -1))[
                    flattened_mask
                ]  # (inside_points, 3)
                rendering_colors_with_alpha = np.hstack(
                    [rendering_colors, 0.5 * np.ones((num_inside_points, 1))]
                )
                occupancy_renderables = [
                    parse_points_to_sphereclouds(
                        rendering_points, rendering_colors_with_alpha, sizes=0.005
                    )
                ]
                tmp_path = f"/tmp/ray_points_occupancy_{id_generator(9)}.png"
                image_from_renderable_list(
                    occupancy_renderables,
                    tmp_path,
                    rendering_scene=rendering_scene,
                    camera_position=camera_pos,
                    up_vector=up_vector,
                    background=(0.0, 0.0, 0.0, 1),
                )
                logger.add_media(
                    epoch,
                    Image(tmp_path),
                    f"ray_points_occupancy_img_{batch_idx:04}_{i:03}",
                )
                if i == 0:
                    tmp_path = f"/tmp/rendered_ray_occupancy_gif_{id_generator(9)}.gif"
                    gif_from_renderable_list(
                        occupancy_renderables,
                        tmp_path,
                        rendering_scene=rendering_scene,
                        camera_position=camera_pos,
                        up_vector=up_vector,
                        background=(0.0, 0.0, 0.0, 1),
                    )
                    logger.add_media(
                        epoch,
                        Video(tmp_path),
                        f"ray_occupancy_video_{batch_idx:04}_000",
                    )

    # Flag to draw per-primitive rendered images
    draw_per_primitive_images = (
        (pred_rgb is not None)
        and (ray_associations is not None)
        and (no_rendering_rays is not None)
    )

    if draw_per_primitive_images:
        for i in range(B):
            pred_rgb_item = pred_rgb[i]  # (N, 3)
            ray_associations_item = ray_associations[i]  # (N,)
            no_rendering_rays_item = no_rendering_rays[i]  # (N,)
            rows_idx_item, cols_idx_item = rows_idx[i], cols_idx[i]
            H_item, W_item = H[i], W[i]
            for j in range(M):
                np_img = np.zeros((H_item, W_item, 3))
                in_primitive = (ray_associations_item == j) * (
                    ~no_rendering_rays_item
                )  # only render when ray is positive
                if in_primitive.sum() > 0:
                    vals = pred_rgb_item[in_primitive, :]
                    rows_idx_j = rows_idx_item[in_primitive]
                    cols_idx_j = cols_idx_item[in_primitive]
                    np_img[rows_idx_j, cols_idx_j] = vals
                pil_img = numpy_images_to_pil_batched([np_img])[0]
                logger.add_media(
                    epoch,
                    Image(pil_img),
                    f"pred_rgb_{batch_idx:04}_{i:03}_primitive_{j:02}",
                )
