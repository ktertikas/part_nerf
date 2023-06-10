from typing import Any, Dict, Iterable, Tuple

import torch
from drawing_utils import add_nerf_primitive_data_to_logger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from train_utils import forward_one_batch, forward_one_batch_coarse_fine
from utils import (
    collect_images_from_keys,
    parse_images_to_logger,
    parse_losses_to_logger,
    parse_metrics_to_logger,
)

from part_nerf.loss import calculate_losses
from part_nerf.metrics import calculate_metrics
from part_nerf.stats_logger import StatsLogger
from part_nerf.utils import send_to, torch_container_to_numpy


def yield_infinite(iterable: Iterable):
    while True:
        for item in iterable:
            yield item


def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    renderer: nn.Module,
    loss_config: Dict,
    metric_config: Dict,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epoch: int,
    steps_per_epoch: int,
    device: torch.device,
    stats_logger: StatsLogger,
    train_data_cfg: Dict,
    visualize_every: int = 10,
    coarse_fine: bool = False,
    grad_accumulation_steps: int = 1,
):
    rays_chunk = train_data_cfg.get("rays_chunk", None)
    num_samples = train_data_cfg.get("n_samples", 32)
    data_type = train_data_cfg.get("collection_type", "lego_dataset")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # model train mode
    model.train()
    # Zero gradients
    optimizer.zero_grad()
    for i, sample in zip(range(steps_per_epoch), yield_infinite(train_loader)):
        X = send_to(sample, device=device)
        # Forward pass and loss calculation
        if coarse_fine:
            predictions = forward_one_batch_coarse_fine(
                model=model,
                renderer=renderer,
                X=X,
                rays_chunk=rays_chunk,
                num_samples=num_samples,
            )
        else:
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
        if (i + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

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
        if i == 0 and (epoch % visualize_every) == 0:
            # Only visualize the first batch
            targets = torch_container_to_numpy(X)
            images_dict = collect_images_from_keys(
                predictions=predictions, targets=targets, keys=["colors", "rgb"]
            )
            parse_images_to_logger(images_dict, stats_logger, batch_idx=i, epoch=epoch)
            add_nerf_primitive_data_to_logger(
                predictions=predictions,
                targets=targets,
                logger=stats_logger,
                batch_idx=i,
                data_type=data_type,
                epoch=epoch,
            )
        stats_logger.print_progress(
            epoch=epoch, batch=i, loss=batch_loss_dict["total_loss"]
        )
    if scheduler is not None:
        # Only log the learning rate for the last step of the epoch, assuming one group
        stats_logger["lr"].value = scheduler.get_last_lr()[0]
    stats_logger.clear()


def validate(
    val_loader: DataLoader,
    model: nn.Module,
    renderer: nn.Module,
    loss_config: Dict,
    metric_config: Dict,
    device: torch.device,
    stats_logger: StatsLogger,
    val_data_cfg: Dict,
    coarse_fine: bool = False,
):
    rays_chunk = val_data_cfg.get("rays_chunk", None)
    num_samples = val_data_cfg.get("n_samples", 32)
    data_type = val_data_cfg.get("collection_type", "lego_dataset")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # model eval mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            X = send_to(sample, device=device)
            loss_dict, metric_dict, preds = validate_one_batch(
                model,
                renderer,
                loss_config,
                metric_config,
                X,
                rays_chunk=rays_chunk,
                num_samples=num_samples,
                coarse_fine=coarse_fine,
            )
            parse_losses_to_logger(loss_dict, stats_logger)
            parse_metrics_to_logger(metric_dict, stats_logger)
            if i == 0:
                # Only visualize the first batch
                targets = torch_container_to_numpy(X)
                images_dict = collect_images_from_keys(
                    predictions=preds,
                    targets=targets,
                    keys=["colors", "rgb", "depth", "disparity"],
                )
                parse_images_to_logger(images_dict, stats_logger, batch_idx=i, epoch=-1)
                add_nerf_primitive_data_to_logger(
                    predictions=preds,
                    targets=targets,
                    logger=stats_logger,
                    batch_idx=i,
                    data_type=data_type,
                    epoch=-1,
                )
            stats_logger.print_progress(epoch=-1, batch=i, loss=loss_dict["total_loss"])
        stats_logger.clear()


def validate_one_batch(
    model: nn.Module,
    renderer: nn.Module,
    loss_config: Dict,
    metric_config: Dict,
    X: Dict,
    rays_chunk: int = None,
    num_samples: int = 32,
    coarse_fine: bool = False,
) -> Tuple[Dict, Dict, Any]:
    # Forward pass and loss calculation
    if coarse_fine:
        predictions = forward_one_batch_coarse_fine(
            model,
            renderer,
            X,
            rays_chunk,
            num_samples=num_samples,
        )
    else:
        predictions = forward_one_batch(model, renderer, X, rays_chunk)

    # Loss and metrics calculations
    batch_loss_dict = calculate_losses(loss_config, predictions=predictions, targets=X)
    metric_dict = calculate_metrics(metric_config, predictions=predictions, targets=X)

    # Detach appropriate structures
    predictions = torch_container_to_numpy(predictions)
    batch_loss_dict = torch_container_to_numpy(batch_loss_dict)
    return batch_loss_dict, metric_dict, predictions
