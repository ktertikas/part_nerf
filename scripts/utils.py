import pickle
import random
import string
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from autodecoder_definitions import (
    InferConfigSchema as NerfAutodecoderInferConfigSchema,
)
from autodecoder_definitions import InversionConfigSchema
from autodecoder_definitions import (
    TrainConfigSchema as NerfAutodecoderTrainConfigSchema,
)
from inference_definitions import EditConfigSchema, InferenceConfigSchema
from matplotlib import cm
from nerf_definitions import InferConfigSchema as NerfInferConfigSchema
from nerf_definitions import TrainConfigSchema as NerfTrainConfigSchema
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from wandb import Image as wandbImage

from part_nerf.stats_logger import StatsLogger

CONFIG_FACTORY = {
    "nerf_train": NerfTrainConfigSchema,
    "nerf_infer": NerfInferConfigSchema,
    "nerf_autodecoder_train": NerfAutodecoderTrainConfigSchema,
    "nerf_autodecoder_infer": NerfAutodecoderInferConfigSchema,
    "nerf_autodecoder_edit": EditConfigSchema,
    "nerf_autodecoder_generation": InferenceConfigSchema,
    "nerf_autodecoder_inversion": InversionConfigSchema,
}


def parse_cli_config() -> Tuple[DictConfig, DictConfig]:
    cli_conf = OmegaConf.from_cli()
    if "config" not in cli_conf:
        raise ValueError(
            "Please provide the 'config' cli flag to specify the yaml file used for running the experiment"
        )
    yaml_conf = OmegaConf.load(cli_conf.config)
    # now remove the config option
    cli_conf.pop("config")
    return yaml_conf, cli_conf


def build_config(conf_type: str) -> DictConfig:
    yaml_conf, cli_conf = parse_cli_config()
    schema_conf = OmegaConf.structured(CONFIG_FACTORY[conf_type])
    config: CONFIG_FACTORY[conf_type] = OmegaConf.merge(
        schema_conf, yaml_conf, cli_conf
    )
    return config


def get_git_commit_hash() -> str:
    try:
        label = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except:
        label = ""
    return label


def id_generator(
    size: int = 6, chars: str = string.ascii_uppercase + string.digits
) -> str:
    return "".join(random.choice(chars) for _ in range(size))


def save_experiment_params(
    config: DictConfig, experiment_name: str, experiment_directory: Path
) -> None:
    git_commit = get_git_commit_hash()

    # appending needed params for reproducibility
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict["trainer"]["git_commit"] = git_commit
    config_dict["trainer"]["experiment_name"] = experiment_name

    # saving params to yaml file
    OmegaConf.save(config_dict, (experiment_directory / "params.yaml"), resolve=True)


def random_seed(
    min_val: int = np.iinfo(np.uint32).min, max_val: int = np.iinfo(np.uint32).max
) -> int:
    return random.randint(min_val, max_val)


def set_all_seeds(seed: int = None):
    if seed is None:
        seed = random_seed()
        print(f"seed not specified, generated new seed number {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def colormap(x, name: str = "hsv", lut: int = 200):
    cmap = cm.get_cmap(name, lut)
    return cmap(x)[:, :3]


def save_pickle(obj: Any, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_latent_codes(latent_path: Path, instance_id: int) -> Tuple:
    # Loading from existing latent codes
    print(f"Loading latent codes of id {instance_id} from {latent_path}")
    shape_latent_path = latent_path / f"shape_{instance_id:04d}"
    texture_latent_path = latent_path / f"texture_{instance_id:04d}"
    assert (
        shape_latent_path.is_file() and texture_latent_path.is_file()
    ), "Latent code does not exist"
    # Load latent code
    shape_code = load_pickle(shape_latent_path)
    texture_code = load_pickle(texture_latent_path)
    return shape_code, texture_code


def parse_metrics_to_logger(metric_dict: Dict, logger: StatsLogger):
    for k, v in metric_dict.items():
        logger[k].value = v


def parse_losses_to_logger(loss_dict: Dict, logger: StatsLogger):
    for k, v in loss_dict.items():
        if k == "total_loss":
            continue
        logger[k].value = v


def parse_images_to_logger(
    images_dict: Dict[str, List[np.ndarray]],
    logger: StatsLogger,
    batch_idx: int,
    epoch: int = -1,
) -> None:
    for img_type, img_list in images_dict.items():
        for i, img in enumerate(img_list):
            logger.add_media(
                epoch, wandbImage(img), f"{img_type}_{batch_idx:04}_{i:03}"
            )


def numpy_images_to_pil_batched(imgs: np.ndarray) -> List[Image.Image]:
    img_list = []
    for img in imgs:
        if img.max() > 1.0:
            img = Image.fromarray((img * 255 / img.max()).round().astype("uint8"))
        else:
            img = Image.fromarray((img * 255).round().astype("uint8"))
        img_list.append(img)
    return img_list


def img_from_values_batched(
    vals: np.ndarray,
    rows_idx: np.ndarray,
    cols_idx: np.ndarray,
    height: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    H = np.unique(height).astype(np.int16)
    W = np.unique(width).astype(np.int16)
    assert len(H) == 1
    assert len(W) == 1
    assert (
        rows_idx.shape[0] == cols_idx.shape[0]
    ), f"Batches for rows and cols indices should be equal"
    H, W = H.item(), W.item()

    # Generate image to be filled
    B = rows_idx.shape[0]
    np_img = np.zeros((B, H, W, 3))
    # Fill image
    np_img[np.arange(B)[:, None], rows_idx, cols_idx] = vals
    return np_img


def collect_images_from_keys(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    keys: List[str] = ["gt_rgb", "pred_rgb"],
) -> Dict[str, List[Image.Image]]:
    H, W = targets["H"], targets["W"]
    rows_idx, cols_idx = targets["sampled_rows"], targets["sampled_cols"]
    images_dict = {}
    for key in keys:
        if key not in predictions.keys() and key not in targets.keys():
            print(f"key {key} not found in predictions or targets, skipping")
            continue
        val = predictions[key] if key in predictions.keys() else targets[key]
        images_dict[key] = numpy_images_to_pil_batched(
            img_from_values_batched(val, rows_idx, cols_idx, H, W)
        )
    return images_dict
