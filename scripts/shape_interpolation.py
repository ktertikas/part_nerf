import argparse
import datetime
import logging
import sys
from pathlib import Path

import torch
from argparse_arguments import add_ray_sampling_args, add_reconstruction_args
from camera_utils import get_camera_origins, get_ray_samples
from editing_utils import (
    add_embedding_info,
    interpolate_vectors,
    run_mesh_extraction_from_part_features,
    run_volumetric_rendering_from_part_features,
)
from inference_definitions import InferenceConfigSchema
from omegaconf import OmegaConf
from reconstruction_utils import MeshGenerator

from part_nerf.model import NerfAutodecoder, build_nerf_autodecoder
from part_nerf.renderer import build_renderer
from part_nerf.utils import dict_to_device_and_batchify, load_checkpoints

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
        experiment_dir = Path(args.checkpoint_path) / f"interpolating_{time_format}"
    experiment_dir.mkdir(exist_ok=True)
    print(f"Saving interpolations on {experiment_dir}")

    # Setup appropriate folders
    full_reconstruction_dir = experiment_dir / f"reconstructions_full"
    part_reconstruction_dir = experiment_dir / f"reconstructions_part"
    images_dir = experiment_dir / f"images"
    full_reconstruction_dir.mkdir(exist_ok=True)
    part_reconstruction_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Set device for interpolation
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Shape Interpolation on device {device}")

    # Model settings
    num_parts = config.model.shape_decomposition_network.num_parts
    # Part interpolation settings
    latent_path_1 = args.latent_path_1
    if latent_path_1 is not None:
        latent_path_1 = Path(latent_path_1)
    embedding_id_1 = args.embedding_id_1
    latent_path_2 = args.latent_path_2
    if latent_path_2 is not None:
        latent_path_2 = Path(latent_path_2)
    embedding_id_2 = args.embedding_id_2
    interpolation_factors = args.interpolation_factors
    # Reconstruction settings
    resolution = args.resolution
    threshold = args.threshold
    mise_resolution = args.mise_resolution
    upsamling_steps = args.upsampling_steps
    padding = args.padding
    with_parts = args.no_parts
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

    # Shape interpolation step
    print("=====> Shape Interpolation =====>")

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
    )

    model.eval()
    with torch.no_grad():
        # Take the first camera origin to render both first and second shape instance
        ray_origin_sample = ray_origins[0]
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

        # Load embeddings from existing saved latent codes or from pretrained model
        shape1_X = add_embedding_info(X, latent_path_1, embedding_id_1, device)
        shape2_X = add_embedding_info(X, latent_path_2, embedding_id_2, device)

        # Inference stage for first shape instance
        print(f"Running inference stage for first sample")
        # Per shape part features
        shape1_predictions = model.forward_part_features(shape1_X)
        # Extract meshes and render one image
        run_mesh_extraction_from_part_features(
            model,
            mesh_generator,
            shape1_predictions,
            full_reconstruction_dir,
            part_reconstruction_dir,
            name="shape1",
            id=embedding_id_1,
            chunk_size=chunk_size,
            device=device,
            num_parts=num_parts,
            with_parts=with_parts,
        )
        run_volumetric_rendering_from_part_features(
            model,
            renderer,
            shape1_X,
            shape1_predictions,
            images_dir,
            rays_chunk,
            id=embedding_id_1,
            name="shape1",
        )

        # Inference stage for second shape instance
        print(f"Running inference stage for second sample")
        shape2_predictions = model.forward_part_features(shape2_X)
        # Extract meshes and render one image
        run_mesh_extraction_from_part_features(
            model,
            mesh_generator,
            shape2_predictions,
            full_reconstruction_dir,
            part_reconstruction_dir,
            name="shape2",
            id=embedding_id_2,
            chunk_size=chunk_size,
            device=device,
            num_parts=num_parts,
            with_parts=with_parts,
        )
        run_volumetric_rendering_from_part_features(
            model,
            renderer,
            shape2_X,
            shape2_predictions,
            images_dir,
            rays_chunk,
            id=embedding_id_2,
            name="shape2",
        )

        # Shape Interpolation stage
        print(f"Running shape interpolation stage")
        shape_embedding_1 = shape1_predictions["shape_embedding"]
        texture_embedding_1 = shape1_predictions["texture_embedding"]
        shape_embedding_2 = shape2_predictions["shape_embedding"]
        texture_embedding_2 = shape2_predictions["texture_embedding"]

        def run_interpolation_combinations(interpolation_factor: float):
            interpolation_percentage = str(round(100 * interpolation_factor))
            new_shape_embedding = interpolate_vectors(
                shape_embedding_1, shape_embedding_2, interpolation_factor
            )
            new_texture_embedding = interpolate_vectors(
                texture_embedding_1, texture_embedding_2, interpolation_factor
            )
            # Only shape interpolation
            shape_predictions_interpolated = model.forward_part_features(
                {
                    "shape_embedding": new_shape_embedding,
                    "texture_embedding": texture_embedding_1,
                }
            )

            # Only texture interpolation
            texture_predictions_interpolated = model.forward_part_features(
                {
                    "shape_embedding": shape_embedding_1,
                    "texture_embedding": new_texture_embedding,
                }
            )

            # Both shape and texture interpolation
            both_predictions_interpolated = model.forward_part_features(
                {
                    "shape_embedding": new_shape_embedding,
                    "texture_embedding": new_texture_embedding,
                }
            )

            # Extracting meshes only once, as texture interpolation does not affect the geometry
            print(
                f"Extracting meshes for shape interpolation with factor {interpolation_factor}"
            )
            run_mesh_extraction_from_part_features(
                model,
                mesh_generator,
                shape_predictions_interpolated,
                full_reconstruction_dir,
                part_reconstruction_dir,
                f"shape_interpolation_{interpolation_percentage}",
                embedding_id_1,
                chunk_size=chunk_size,
                device=device,
                num_parts=num_parts,
                with_parts=with_parts,
            )

            print(f"Extracting images for {num_views} views")
            for i in range(num_views):
                # Volumetric rendering for all cases
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
                # Shape interpolation
                run_volumetric_rendering_from_part_features(
                    model,
                    renderer,
                    X,
                    shape_predictions_interpolated,
                    images_dir,
                    rays_chunk,
                    embedding_id_1,
                    name=f"shape_interpolation_{i:03d}_{interpolation_percentage}",
                )
                # Texture interpolation
                run_volumetric_rendering_from_part_features(
                    model,
                    renderer,
                    X,
                    texture_predictions_interpolated,
                    images_dir,
                    rays_chunk,
                    embedding_id_1,
                    name=f"texture_interpolation_{i:03d}_{interpolation_percentage}",
                )
                # Both
                run_volumetric_rendering_from_part_features(
                    model,
                    renderer,
                    X,
                    both_predictions_interpolated,
                    images_dir,
                    rays_chunk,
                    embedding_id_1,
                    name=f"full_interpolation_{i:03d}_{interpolation_percentage}",
                )

        for interp in interpolation_factors:
            run_interpolation_combinations(interp)


def parse_shape_interpolation_args(argv):
    parser = argparse.ArgumentParser(
        description="Interpolation between two shape instances."
    )
    parser.add_argument(
        "config_file", help="Path to the file that contains the model definition"
    )
    # Interpolation args
    parser.add_argument(
        "--embedding_id_1",
        type=int,
        required=True,
        help="The embedding id of the first shape to be used for interpolation. If the latent path is specified, this argument specifies the saved latent code id",
    )
    parser.add_argument(
        "--latent_path_1",
        type=str,
        default=None,
        help="The path to the latent codes of the first shape used for interpolation",
    )
    parser.add_argument(
        "--embedding_id_2",
        type=int,
        required=True,
        help="The embedding id of the second shape to be used for interpolation. If the latent path is specified, this argument specifies the saved latent code id",
    )
    parser.add_argument(
        "--latent_path_2",
        type=str,
        default=None,
        help="The path to the latent codes of the second shape used for interpolation",
    )
    parser.add_argument(
        "--interpolation_factors",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8],
        help="The interpolation factors used for the shape interpolation",
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
        help="The output path where the interpolations will be stored",
    )
    # Reconstruction args
    parser = add_reconstruction_args(parser)
    parser.add_argument(
        "--no_parts",
        action="store_false",
        help="If option is selected no part reconstructions will happen",
    )
    # Volumetric rendering args
    parser = add_ray_sampling_args(parser)
    parser.add_argument(
        "--num_views",
        type=int,
        default=1,
        help="Number of generated image views for each interpolated shape",
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
    args = parse_shape_interpolation_args(sys.argv[1:])
    main(args)
