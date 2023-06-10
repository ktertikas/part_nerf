# This script takes a model id as input and optimizes an embedding vector
# to match the shape as best as possible.
import datetime
import logging
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Tuple

import torch
from autodecoder_definitions import InversionConfigSchema
from camera_utils import get_camera_origins, get_ray_samples
from nerf_train_utils import calculate_losses, calculate_metrics, yield_infinite
from omegaconf import OmegaConf
from reconstruction_utils import MeshGenerator, reconstruct_meshes_from_model
from torch.utils.data import Subset
from train_utils import forward_one_batch
from trimesh.exchange.export import export_mesh
from utils import (
    build_config,
    img_from_values_batched,
    numpy_images_to_pil_batched,
    parse_losses_to_logger,
    parse_metrics_to_logger,
    save_pickle,
)

from part_nerf.dataset import build_dataloader, build_dataset
from part_nerf.model import NerfAutodecoder, build_nerf_autodecoder
from part_nerf.optimizer import build_optimizer
from part_nerf.renderer import build_renderer
from part_nerf.stats_logger import StatsLogger
from part_nerf.utils import (
    dict_to_device_and_batchify,
    load_checkpoints,
    send_to,
    torch_container_to_numpy,
)

# A logger for this file
logger = logging.getLogger(__name__)
# Disable trimesh's logger
logging.getLogger("trimesh").setLevel(logging.ERROR)


def init_embeddings(
    model: NerfAutodecoder, max_norm: float = None
) -> Tuple[torch.nn.Embedding, torch.nn.Embedding]:
    shape_vector = model.get_random_shape_embeddings(1)
    texture_vector = model.get_random_texture_embeddings(1)
    shape_embedding = torch.nn.Embedding(1, shape_vector.shape[1], max_norm=max_norm)
    texture_embedding = torch.nn.Embedding(
        1, texture_vector.shape[1], max_norm=max_norm
    )
    return shape_embedding, texture_embedding


def main():
    config: InversionConfigSchema = build_config(conf_type="nerf_autodecoder_inversion")
    print(f"Configuration: {OmegaConf.to_yaml(config)}")

    # Specify experiment directory
    experiment_dir = Path(config.invertor.experiment_directory)
    if not experiment_dir.exists():
        raise NotADirectoryError(f"Output directory {experiment_dir} does not exist!")

    # Inversion settings
    num_iters = config.invertor.num_iters
    with_texture = config.invertor.with_texture
    grad_accumulation_steps = config.invertor.grad_accumulation_steps
    rays_chunk = config.data.rays_chunk
    # Reconstruction settings
    resolution = config.invertor.resolution
    threshold = config.invertor.mcubes_threshold
    upsamling_steps = config.invertor.upsampling_steps
    mise_resolution = config.invertor.mise_resolution
    chunk_size = config.model.occupancy_network.chunk_size
    padding = config.invertor.reconstruction_padding

    # Setup reconstruction directories
    time_format = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(experiment_dir)
    inversion_dir = experiment_dir / f"inversion_{time_format}"
    inversion_dir.mkdir(exist_ok=True)
    full_reconstruction_dir = inversion_dir / f"reconstructions_full"
    part_reconstruction_dir = inversion_dir / f"reconstructions_part"
    if with_texture:
        images_dir = inversion_dir / f"images"
        images_dir.mkdir(exist_ok=True)
    latent_code_dir = inversion_dir / f"latents_dir"
    latent_code_dir.mkdir(exist_ok=True)
    full_reconstruction_dir.mkdir(exist_ok=True)
    part_reconstruction_dir.mkdir(exist_ok=True)
    # saving to yaml file for bookkeeping
    OmegaConf.save(config, (inversion_dir / "params.yaml"), resolve=True)

    # Set device for training
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Inverting on device {device}")

    # building dataset, make sure the config file has the correct number of views!
    dataset = build_dataset(
        config.data,
        data_tags=OmegaConf.to_container(config.data.data_tags, resolve=True),
        category_tags=OmegaConf.to_container(config.data.category_tags, resolve=True),
        keep_splits=OmegaConf.to_container(config.data.splits, resolve=True),
        random_subset_pct=config.data.random_subset_pct,
    )

    # find number of views from length of dataset divided by number of instances
    shape_instances_dict = OrderedDict.fromkeys(
        dataset._datasets[0].internal_data_collection[i]._scene
        for i in range(len(dataset))
    )
    shape_instances = list(shape_instances_dict.keys())
    print(f"Number of test shape instances: {len(shape_instances)}")
    num_views = int(len(dataset) / len(shape_instances))
    print(f"Running inversion on {num_views} views")

    model: NerfAutodecoder = build_nerf_autodecoder(config.model)
    model.to(device)
    renderer = build_renderer(config.renderer)

    # Load checkpoint of model
    load_checkpoints(model, None, experiment_dir, config, device, scheduler=None)

    # Setting model parameters to not require gradients.
    for param in model.parameters():
        param.requires_grad = False

    # TRAINING config
    loss_config = OmegaConf.to_container(config.loss, resolve=True)
    metric_config = OmegaConf.to_container(config.metric, resolve=True)

    if with_texture:
        # TODO: Generalize for different camera distances
        ray_origins = get_camera_origins(
            distance=1.5,
            azimuth_start=60,
            azimuth_stop=301,
            azimuth_step=120,
            elevation_start=30,
            elevation_stop=-31,
            elevation_step=-30,
        )
    # Setting shape inversion for all shape instances separately
    for i, model_id in enumerate(shape_instances):
        # select subset depending on number of views used
        current_subset_ids = list(range(num_views * i, num_views * (i + 1)))
        dataset_subset = Subset(dataset, current_subset_ids)
        dl = build_dataloader(
            dataset=dataset_subset,
            batch_size=config.invert.batch_size,
            num_workers=config.invert.num_workers,
            shuffle=config.invert.shuffle,
            pin_memory=config.invert.pin_memory,
        )
        # Get embedding modules to be used in the optimizer
        shape_embedding, texture_embedding = init_embeddings(
            model, config.model.shape_embedding_network.max_norm
        )
        # Stage 1, invert using shape mask - freeze everything, and pass
        # random embedding vector for shape
        shape_embedding = shape_embedding.to(device)
        texture_embedding = texture_embedding.to(device)
        if with_texture:
            optimizer = build_optimizer(
                config.optimizer,
                list(shape_embedding.parameters())
                + list(texture_embedding.parameters()),
            )
        else:
            optimizer = build_optimizer(config.optimizer, shape_embedding.parameters())

        # stats logger init
        stats_logger = StatsLogger()

        # Shape inversion stage
        model.train()
        optimizer.zero_grad()
        for j, sample in zip(range(num_iters), yield_infinite(dl)):
            X = send_to(sample, device=device)
            B = X["ray_directions"].shape[0]
            embedding_idx = torch.zeros(
                [
                    B,
                ],
                dtype=torch.long,
                device=device,
            )
            shape_embedding_samples = shape_embedding(embedding_idx)
            texture_embedding_samples = texture_embedding(embedding_idx)

            # setting shape and texture embeddings
            X["shape_embedding"] = shape_embedding_samples
            X["texture_embedding"] = texture_embedding_samples
            predictions = forward_one_batch(
                model=model,
                renderer=renderer,
                X=X,
                rays_chunk=rays_chunk,
            )
            # loss calculation
            batch_loss_dict = calculate_losses(
                loss_config, predictions=predictions, targets=X
            )
            # Backward pass normalized by grad_accumulation_steps
            (batch_loss_dict["total_loss"] / grad_accumulation_steps).backward()

            # When reaching grad_accumulation_steps, do optimizer.step()
            if (j + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Loss logging
            batch_loss_dict = torch_container_to_numpy(batch_loss_dict)
            parse_losses_to_logger(batch_loss_dict, stats_logger)
            # Metrics calculation and logging
            if metric_config:
                metric_dict = calculate_metrics(
                    metric_config, predictions=predictions, targets=X
                )
                metric_dict = torch_container_to_numpy(metric_dict)
                parse_metrics_to_logger(metric_dict, stats_logger)

            # Detach appropriate structures
            predictions = torch_container_to_numpy(predictions)
            # Parse results
            stats_logger.print_progress(
                epoch=0, batch=j, loss=batch_loss_dict["total_loss"]
            )

        # Inference stage now, reconstruct shape and (maybe) texture!
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Finished shape inversion, inference stage now!")
        # Defining mesh generator object
        mesh_generator = MeshGenerator(
            resolution=resolution,
            mise_resolution=mise_resolution,
            padding=padding,
            threshold=threshold,
            upsampling_steps=upsamling_steps,
        )

        model.eval()
        shape_embedding.eval()
        texture_embedding.eval()

        # Reconstruction stage here
        new_dl = build_dataloader(
            dataset=dataset_subset,
            batch_size=1,
            num_workers=config.invert.num_workers,
            shuffle=config.invert.shuffle,
            pin_memory=config.invert.pin_memory,
        )
        sample = next(iter(new_dl))
        with torch.no_grad():
            embedding_idx = torch.zeros(
                [
                    B,
                ],
                dtype=torch.long,
                device=device,
            )
            shape_embedding_samples = shape_embedding(embedding_idx)
            texture_embedding_samples = texture_embedding(embedding_idx)

            if with_texture:
                # Fixed camera origin
                origin_id = 2
                ray_origin_sample = ray_origins[origin_id]
                # Sample points along the rays
                new_X = get_ray_samples(
                    ray_origin=ray_origin_sample,
                    H=config.data.image_size[0],
                    W=config.data.image_size[1],
                    near=config.data.near,
                    far=config.data.far,
                    num_samples=config.data.n_samples,
                )
                new_X = dict_to_device_and_batchify(new_X, device=device)
                new_X["shape_embedding"] = shape_embedding_samples
                new_X["texture_embedding"] = texture_embedding_samples

                new_predictions = forward_one_batch(
                    model=model, renderer=renderer, X=new_X, rays_chunk=2048
                )
                new_predictions = torch_container_to_numpy(new_predictions)
                detached_targets = torch_container_to_numpy(new_X)
                np_img = img_from_values_batched(
                    new_predictions["rgb"],
                    detached_targets["sampled_rows"],
                    detached_targets["sampled_cols"],
                    detached_targets["H"],
                    detached_targets["W"],
                )
                # Save image
                pil_img = numpy_images_to_pil_batched(np_img)[0]
                pil_img.save((images_dir / f"{model_id}.png"))
                del new_X
                del new_predictions

            X = send_to(sample, device=device)
            X["shape_embedding"] = shape_embedding_samples
            X["texture_embedding"] = texture_embedding_samples
            predictions = model.forward_part_features_and_params(X)
            model_occupancy_callable = partial(
                model.forward_occupancy_field_from_part_preds, pred_dict=predictions
            )
            mesh, part_meshes_list = reconstruct_meshes_from_model(
                model_occupancy_callable,
                mesh_generator,
                chunk_size,
                device=device,
                with_parts=True,
                num_parts=config.model.shape_decomposition_network.num_parts,
            )
        # exporting meshes
        export_mesh(mesh, (full_reconstruction_dir / f"{model_id}.obj"))
        for k, part_mesh in enumerate(part_meshes_list):
            export_mesh(
                part_mesh,
                part_reconstruction_dir / f"{model_id}_{k:02}.obj",
            )
        # save shape and texture embeddings to file for reuse
        shape_embedding = predictions["shape_embedding"].detach().cpu()
        save_pickle(shape_embedding, (latent_code_dir / f"{model_id}_shape"))
        if with_texture:
            texture_embedding = predictions["texture_embedding"].detach().cpu()
            save_pickle(texture_embedding, (latent_code_dir / f"{model_id}_texture"))


if __name__ == "__main__":
    main()
