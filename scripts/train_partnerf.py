import logging
from pathlib import Path

import torch
from autodecoder_definitions import TrainConfigSchema
from nerf_train_utils import train_one_epoch, validate
from omegaconf import OmegaConf
from utils import build_config, id_generator, save_experiment_params, set_all_seeds

from part_nerf.dataset import build_dataloader, build_dataset
from part_nerf.model import build_nerf_autodecoder
from part_nerf.optimizer import build_optimizer
from part_nerf.renderer import build_renderer
from part_nerf.scheduler import build_scheduler
from part_nerf.stats_logger import StatsLogger, WandB
from part_nerf.utils import load_checkpoints, save_checkpoints

# A logger for this file
logger = logging.getLogger(__name__)
# Disable trimesh's logger
logging.getLogger("trimesh").setLevel(logging.ERROR)


def main():
    config: TrainConfigSchema = build_config(conf_type="nerf_autodecoder_train")
    print(f"Configuration: {OmegaConf.to_yaml(config)}")

    # Create output directory
    output_dir = Path(config.trainer.output_directory)
    output_dir.mkdir(exist_ok=True)

    # Create experiment directory
    dummy_id = id_generator(9)
    experiment_name = (
        config.trainer.get("experiment_name", dummy_id) or dummy_id
    )  # in case experiment_name is already None
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    save_experiment_params(config, experiment_name, experiment_dir)

    # Set seed - Important to do this after id generation, as we set seed to random as well!
    set_all_seeds(config.trainer.seed)
    # Set device for training
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Training on device {device}")

    train_dataset = build_dataset(
        config.train_data,
        data_tags=OmegaConf.to_container(config.train_data.data_tags, resolve=True),
        category_tags=OmegaConf.to_container(
            config.train_data.category_tags, resolve=True
        ),
        keep_splits=OmegaConf.to_container(config.train_data.splits, resolve=True),
        random_subset_pct=config.train_data.random_subset_pct,
    )
    train_dl = build_dataloader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=config.train.shuffle,
        pin_memory=config.train.pin_memory,
    )

    if config.validation:
        val_every = config.trainer.run_validation_every
        val_dataset = build_dataset(
            config.val_data,
            data_tags=OmegaConf.to_container(config.val_data.data_tags, resolve=True),
            category_tags=OmegaConf.to_container(
                config.val_data.category_tags, resolve=True
            ),
            keep_splits=OmegaConf.to_container(config.val_data.splits, resolve=True),
            random_subset_pct=config.val_data.random_subset_pct,
        )
        val_dl = build_dataloader(
            dataset=val_dataset,
            batch_size=config.validation.batch_size,
            num_workers=config.validation.num_workers,
            shuffle=config.validation.shuffle,
            pin_memory=config.validation.pin_memory,
        )
    model = build_nerf_autodecoder(config.model)
    model.to(device)
    renderer = build_renderer(config.renderer)
    optimizer = build_optimizer(config.optimizer, model.parameters())
    scheduler = None
    if config.scheduler is not None:
        scheduler = build_scheduler(config.scheduler, optimizer)
    # Load checkpoints if they exist in the experiment_directory
    load_checkpoints(
        model, optimizer, experiment_dir, config, device, scheduler=scheduler
    )

    # TRAINING config
    loss_config = OmegaConf.to_container(config.loss, resolve=True)
    metric_config = OmegaConf.to_container(config.metric, resolve=True)
    start_epoch = config.trainer.get("start_epoch", 0)
    num_epochs = config.trainer.num_epochs
    steps_per_epoch = config.trainer.steps_per_epoch
    save_every = config.trainer.save_checkpoint_every
    train_visualize_every = config.trainer.get("train_visualize_every", 10)

    # stats logger init
    if config.trainer.statslogger == "wandb":
        stats_logger = WandB(
            OmegaConf.to_container(config, resolve=True),
            model,
            project=config.trainer.project_name,
            name=experiment_name,
            experiment_dir=experiment_dir,
            start_epoch=start_epoch,
        )
    else:
        # simple stats logger
        stats_logger = StatsLogger()

    # TRAINING step
    for epoch in range(start_epoch, num_epochs):
        train_one_epoch(
            train_dl,
            model,
            renderer,
            loss_config,
            metric_config,
            optimizer,
            scheduler,
            epoch,
            steps_per_epoch,
            device,
            stats_logger,
            config.train_data,
            visualize_every=train_visualize_every,
            coarse_fine=config.model.coarse_fine,
            grad_accumulation_steps=config.trainer.grad_accumulation_steps,
        )
        save_checkpoint = (
            save_every
            and (epoch % save_every) == 0
            and epoch != 0
            or epoch == num_epochs - 1
        )
        if save_checkpoint:
            save_checkpoints(
                epoch, model, optimizer, experiment_dir, scheduler=scheduler
            )

        run_validation = config.validation and (
            ((epoch % val_every) == 0 and (epoch > 0)) or epoch == num_epochs - 1
        )
        if run_validation:
            print("=====> Validation Epoch Start =====>")
            validate(
                val_dl,
                model,
                renderer,
                loss_config,
                metric_config,
                device,
                stats_logger,
                config.val_data,
                coarse_fine=config.model.coarse_fine,
            )
            print("=====> Validation Epoch End =======>")


if __name__ == "__main__":
    main()
