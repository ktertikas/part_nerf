import argparse
import datetime
import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from argparse_arguments import add_ray_sampling_args, add_reconstruction_args
from camera_utils import get_camera_origins, get_ray_samples
from editing_utils import add_embedding_info
from inference_definitions import InferenceConfigSchema
from omegaconf import OmegaConf
from pyquaternion import Quaternion
from reconstruction_utils import (
    MeshGenerator,
    export_meshes_to_path,
    reconstruct_meshes_from_model,
)
from train_utils import forward_one_batch
from utils import collect_images_from_keys

from part_nerf.model import NerfAutodecoder, build_nerf_autodecoder
from part_nerf.renderer import build_renderer
from part_nerf.utils import (
    dict_to_device_and_batchify,
    load_checkpoints,
    torch_container_to_numpy,
)

# A logger for this file
logger = logging.getLogger(__name__)
# Disable trimesh's logger
logging.getLogger("trimesh").setLevel(logging.ERROR)


def main(args):
    # model config validation
    yaml_conf = OmegaConf.load(args.config_file)
    schema_conf = OmegaConf.structured(InferenceConfigSchema)
    config: InferenceConfigSchema = OmegaConf.merge(schema_conf, yaml_conf)
    print(f"Configuration: {OmegaConf.to_yaml(config)}")

    # Specify experiment directory
    if args.output_path is not None:
        experiment_dir = Path(args.output_path)
    else:
        # specify time in order to be able to differentiate folders
        time_format = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path(args.checkpoint_path) / f"editing_{time_format}"
    experiment_dir.mkdir(exist_ok=True)
    print(f"Saving edits on {experiment_dir}")

    # Setup appropriate folders
    full_reconstruction_dir = experiment_dir / f"reconstructions_full"
    part_reconstruction_dir = experiment_dir / f"reconstructions_part"
    images_dir = experiment_dir / f"images"
    full_reconstruction_dir.mkdir(exist_ok=True)
    part_reconstruction_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Set device for editing
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Editing on device {device}")

    # Model settings
    num_parts = config.model.shape_decomposition_network.num_parts
    # Editing settings
    latent_path = args.latent_path
    if latent_path is not None:
        latent_path = Path(latent_path)
    embedding_id = args.embedding_id
    part_ids = args.part_ids
    # Reconstruction settings
    resolution = args.resolution
    threshold = args.threshold
    mise_resolution = args.mise_resolution
    upsamling_steps = args.upsampling_steps
    padding = args.padding
    chunk_size = config.model.occupancy_network.chunk_size
    checkpoint_id = args.checkpoint_id
    checkpoint_path = Path(args.checkpoint_path)
    # Rendering args
    num_views = args.num_views
    camera_distance = args.camera_distance
    near = args.near
    far = args.far
    H = args.height
    W = args.width
    up_vector = args.up_vector
    num_samples = args.num_point_samples
    rays_chunk = args.rays_chunk

    model: NerfAutodecoder = build_nerf_autodecoder(config.model)
    model.to(device)
    renderer = build_renderer(config.renderer)
    # Load checkpoints if they exist in the experiment_directory
    load_checkpoints(
        model,
        None,
        checkpoint_path,
        config,
        device,
        model_id=checkpoint_id,
    )

    # Editing step
    print("=====> Editing Suite Start =====>")

    # Defining mesh generator object
    mesh_generator = MeshGenerator(
        resolution=resolution,
        mise_resolution=mise_resolution,
        padding=padding,
        threshold=threshold,
        upsampling_steps=upsamling_steps,
    )

    # defining views along the same elevation level, different azimuth
    azimuth_step = 360 // num_views
    ray_origins = get_camera_origins(
        distance=camera_distance,
        azimuth_start=30,
        azimuth_stop=389,
        azimuth_step=azimuth_step,
        elevation_start=10,
        elevation_stop=0,
        elevation_step=-11,
        up=up_vector,
    )

    model.eval()
    with torch.no_grad():
        # Take the first camera origin
        ray_origin_sample = ray_origins[0]
        # Sample points along the rays
        X = get_ray_samples(
            ray_origin=ray_origin_sample,
            H=H,
            W=W,
            near=near,
            far=far,
            num_samples=num_samples,
            up=up_vector,
        )
        X = dict_to_device_and_batchify(X, device=device)
        # Load latent code or specify embedding id
        X = add_embedding_info(X, latent_path, embedding_id, device)

        # Inference stage
        print(f"Running inference stage for selected sample")
        predictions = forward_one_batch(model, renderer, X, rays_chunk=rays_chunk)
        detached_predictions = torch_container_to_numpy(predictions)
        detached_targets = torch_container_to_numpy(X)
        images = collect_images_from_keys(
            predictions=detached_predictions,
            targets=detached_targets,
            keys=["rgb"],
        )
        images["rgb"][0].save((images_dir / f"original_{embedding_id:04d}.png"))

        # Extract and export mesh
        model_occupancy_callable = partial(
            model.forward_occupancy_field_from_part_preds, pred_dict=predictions
        )
        mesh, part_meshes_list = reconstruct_meshes_from_model(
            model_occupancy_callable,
            mesh_generator,
            chunk_size,
            device,
            with_parts=True,
            num_parts=num_parts,
        )
        export_meshes_to_path(
            full_reconstruction_dir,
            part_reconstruction_dir,
            embedding_id,
            mesh,
            part_meshes_list,
        )

        # Editing stage - Affine transformations only
        print(f"Running editing stage for selected sample")
        translation_vector = (
            torch.tensor(args.translation, device=device)
            if args.translation is not None
            else torch.zeros(3, device=device)
        )
        rotation_quaternion = Quaternion(
            axis=args.rotation_axis, angle=np.deg2rad(args.rotation_angle)
        )

        # Apply transformations, first rotation, then translation
        part_translations = predictions["translations"]
        part_rotations = predictions["rotations"]
        predictions_affine = predictions.copy()
        for pid in part_ids:
            q = Quaternion(part_rotations[0, pid, :].cpu().numpy())
            r = q * rotation_quaternion
            new_r = r.elements
            predictions_affine["rotations"][0, pid, :] = torch.tensor(
                new_r, device=device
            )

        new_translations = torch.zeros_like(part_translations)
        new_translations[:, part_ids, :] = translation_vector
        new_translations += part_translations
        predictions_affine["translations"] = new_translations

        # Extract and export mesh
        model_occupancy_callable = partial(
            model.forward_occupancy_field_from_part_preds, pred_dict=predictions_affine
        )
        mesh, part_meshes_list = reconstruct_meshes_from_model(
            model_occupancy_callable,
            mesh_generator,
            chunk_size,
            device,
            with_parts=True,
            num_parts=num_parts,
        )
        export_meshes_to_path(
            full_reconstruction_dir,
            part_reconstruction_dir,
            embedding_id,
            mesh,
            part_meshes_list,
            name="edited",
        )

        # Get Image Reconstructions
        model_color_callable = partial(
            model.forward_color_field_from_part_preds, pred_dict=predictions_affine
        )
        for i in range(num_views):
            # Take the first camera origin
            ray_origin_sample = ray_origins[i]
            # Sample points along the rays
            X = get_ray_samples(
                ray_origin=ray_origin_sample,
                H=H,
                W=W,
                near=near,
                far=far,
                num_samples=num_samples,
            )
            X = dict_to_device_and_batchify(X, device=device)
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
            images["rgb"][0].save((images_dir / f"edited_{i:04}.png"))
            del color_predictions


def parse_shape_editing_args(argv):
    parser = argparse.ArgumentParser(
        description="Editing of one or multiple parts from a single shape instance."
    )
    parser.add_argument(
        "config_file", help="Path to the file that contains the model definition"
    )
    # Editing args
    parser.add_argument(
        "--part_ids",
        type=int,
        nargs="+",
        required=True,
        help="The part ids that will be edited",
    )
    parser.add_argument(
        "--embedding_id",
        type=int,
        required=True,
        help="The embedding id of the shape to be used for editing. If the latent path is specified, this argument specifies the saved latent code id",
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        default=None,
        help="The path to the latent codes of the shape used for editing",
    )
    parser.add_argument(
        "--translation",
        type=float,
        nargs=3,
        default=None,
        help="The translation vector to transform the selected parts",
    )
    parser.add_argument(
        "--rotation_axis",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        help="The rotation axis in which we rotate the selected parts",
    )
    parser.add_argument(
        "--rotation_angle",
        type=float,
        default=0.0,
        help="The rotation angle in degrees used to rotate the selected parts",
    )
    # Experiment args
    parser.add_argument(
        "--checkpoint_id",
        type=int,
        default=None,
        help="The checkpoint id number. If not specified the script loads the last checkpoint in the experiment folder",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="The base path where the checkpoint exists. If an output directory is not specified, the checkpoint path will be used",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The output path where the edits will be stored",
    )
    # Reconstruction args
    parser = add_reconstruction_args(parser)
    # Volumetric rendering args
    parser = add_ray_sampling_args(parser)
    parser.add_argument(
        "--num_views",
        type=int,
        default=1,
        help="Number of generated image views for each shape",
    )
    parser.add_argument(
        "--rays_chunk",
        type=int,
        default=512,
        help="Ray chunk size",
    )
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parse_shape_editing_args(sys.argv[1:])
    main(args)
