import argparse
import datetime
import logging
import sys
from functools import partial
from pathlib import Path

import torch
from argparse_arguments import add_ray_sampling_args, add_reconstruction_args
from camera_utils import get_camera_origins, get_ray_samples
from inference_definitions import InferenceConfigSchema
from omegaconf import OmegaConf
from reconstruction_utils import MeshGenerator, reconstruct_meshes_from_model
from train_utils import forward_one_batch
from trimesh.exchange.export import export_mesh
from utils import collect_images_from_keys, save_pickle

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
        experiment_dir = Path(args.checkpoint_path) / f"generating_{time_format}"
    experiment_dir.mkdir(exist_ok=True)
    print(f"Saving generations on {experiment_dir}")
    # Setup appropriate folders
    full_reconstruction_dir = experiment_dir / f"reconstructions_full"
    part_reconstruction_dir = experiment_dir / f"reconstructions_part"
    latent_code_dir = experiment_dir / f"latents_dir"
    images_dir = experiment_dir / f"images"
    full_reconstruction_dir.mkdir(exist_ok=True)
    part_reconstruction_dir.mkdir(exist_ok=True)
    latent_code_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Generation args
    num_generations = args.num_generations
    resolution = args.resolution
    threshold = args.threshold
    mise_resolution = args.mise_resolution
    upsamling_steps = args.upsampling_steps
    padding = args.padding
    with_parts = args.no_parts
    checkpoint_id = args.checkpoint_id
    checkpoint_path = Path(args.checkpoint_path)
    chunk_size = config.model.occupancy_network.chunk_size

    # Rendering args
    with_renders = args.with_renders
    num_views = args.num_views
    camera_distance = args.camera_distance
    near = args.near
    far = args.far
    H = args.height
    W = args.width
    num_samples = args.num_point_samples
    rays_chunk = args.rays_chunk

    # Set device for generation
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Generation on device {device}")

    model: NerfAutodecoder = build_nerf_autodecoder(config.model)
    model.to(device)

    if with_renders:
        renderer = build_renderer(config.renderer)
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

    # Load checkpoints if they exist in the experiment_directory
    load_checkpoints(
        model,
        None,
        checkpoint_path,
        config,
        device,
        model_id=checkpoint_id,
    )

    # Generation step
    print("=====> Generation Start =====>")

    # Defining mesh generator object
    mesh_generator = MeshGenerator(
        resolution=resolution,
        mise_resolution=mise_resolution,
        padding=padding,
        threshold=threshold,
        upsampling_steps=upsamling_steps,
    )
    # model eval mode
    model.eval()
    with torch.no_grad():
        i = 0
        while i < num_generations:
            # Forward random latent and get primitive parameters
            predictions = model.forward_part_features_and_params_from_random(
                num_embeddings=1
            )

            # Run implicit field and extract mesh
            model_occupancy_callable = partial(
                model.forward_occupancy_field_from_part_preds,
                pred_dict=predictions,
            )

            try:
                mesh, part_meshes_list = reconstruct_meshes_from_model(
                    model_occupancy_callable,
                    mesh_generator,
                    chunk_size,
                    device,
                    with_parts=with_parts,
                    num_parts=config.model.shape_decomposition_network.num_parts,
                )
            except Exception as e:
                print("Latent code produced mesh generation error, skipping...")
                continue
            if mesh.is_empty:
                print("Generated mesh is empty, skipping...")
                continue

            # Export meshes
            mesh_path = full_reconstruction_dir / f"full_mesh_{i:04}.obj"
            export_mesh(mesh, mesh_path)
            for j, part_mesh in enumerate(part_meshes_list):
                export_mesh(
                    part_mesh,
                    (part_reconstruction_dir / f"part_mesh_{i:04}_{j:02}.obj"),
                )

            # Render images if selected
            if with_renders:
                for j in range(num_views):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # TODO: Refactor into a reusable component!
                    # Get ray samples for evaluation
                    camera_origin = ray_origins[j]
                    X = get_ray_samples(
                        ray_origin=camera_origin,
                        H=H,
                        W=W,
                        near=near,
                        far=far,
                        num_samples=num_samples,
                    )
                    X = dict_to_device_and_batchify(X, device=device)
                    # Run implicit field and extract mesh
                    model_color_callable = partial(
                        model.forward_color_field_from_part_preds,
                        pred_dict=predictions,
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
                    images["rgb"][0].save((images_dir / f"img_{i:04}_{j:03}.png"))
                    del color_predictions

            # save shape and texture embeddings to file for reuse
            shape_embedding = predictions["shape_embedding"].detach().cpu()
            save_pickle(shape_embedding, (latent_code_dir / f"shape_{i:04}"))
            texture_embedding = predictions["texture_embedding"].detach().cpu()
            save_pickle(texture_embedding, (latent_code_dir / f"texture_{i:04}"))

            i += 1


def parse_shape_generation_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate new shapes using a pretrained model."
    )
    parser.add_argument(
        "config_file", help="Path to the file that contains the model definition"
    )
    parser.add_argument(
        "--num_generations", type=int, default=100, help="Number of generated shapes"
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
        "--with_renders",
        action="store_true",
        help="If selected renderings of the shape will be generated",
    )
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
        required=False,
        default=None,
        help="The output path where the generations will be stored",
    )

    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parse_shape_generation_args(sys.argv[1:])
    main(args)
