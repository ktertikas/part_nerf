"""Script that defines the input argument types coming from the
configuration files for NeRF training."""
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class Data:
    dataset_factory: str = MISSING
    collection_type: str = MISSING
    dataset_directory: str = MISSING
    images_folder: str = "images"
    cameras_folder: str = "cameras"
    masks_folder: str = "masks"
    white_background: bool = False
    splits_file: str = MISSING
    data_tags: List = field(default_factory=lambda: [])
    category_tags: List = field(default_factory=lambda: [])
    random_subset_pct: float = 1.0
    image_size: List[int] = field(default_factory=lambda: [100, 100])
    n_samples: int = 64
    rand: bool = True
    near: float = 2.0
    far: float = 6.0
    perturb: float = 0.0
    rays_chunk: Optional[int] = None


@dataclass
class TrainData(Data):
    n_rays: int = MISSING
    sampling_type: Optional[str] = "uniform"
    splits: List[str] = field(default_factory=lambda: ["train", "val"])


@dataclass
class ValData(Data):
    n_rays: int = -1
    sampling_type: Optional[str] = None
    splits: List[str] = field(default_factory=lambda: ["test"])


@dataclass
class Train:
    batch_size: int = 4
    num_workers: int = 6
    shuffle: bool = True
    pin_memory: bool = True


@dataclass
class Validation:
    batch_size: int = 1
    num_workers: int = 2
    shuffle: bool = False
    pin_memory: bool = True


@dataclass
class FeatureExtractor:
    type: str = "mlp"
    input_dims: int = 319
    out_dims: Optional[int] = -1
    proj_dims: List = field(default_factory=lambda: [256, 256, 256, 256])
    activation: str = "relu"
    last_activation: Optional[str] = None


@dataclass
class PointEncoder:
    type: str = "mlp"
    input_dims: int = 63
    out_dims: Optional[int] = -1
    proj_dims: List[int] = field(default_factory=lambda: [256, 256, 256, 256, 256])
    activation: str = "relu"
    last_activation: Optional[str] = "relu"


@dataclass
class ColorEncoder:
    type: str = "mlp"
    input_dims: int = 259
    out_dims: Optional[int] = -1
    proj_dims: List[int] = field(default_factory=lambda: [128, 3])
    activation: str = "relu"
    last_activation: Optional[str] = "sigmoid"


@dataclass
class DensityEncoder:
    type: Optional[str] = None


@dataclass
class Model:
    type: str = "nerf"
    feature_extractor: FeatureExtractor = FeatureExtractor()
    point_encoder: PointEncoder = PointEncoder()
    color_encoder: ColorEncoder = ColorEncoder()
    density_encoder: DensityEncoder = DensityEncoder()


@dataclass
class Renderer:
    type: str = "nerf"
    background_opacity: float = 1e10
    density_noise_std: float = 0.0
    white_background: bool = False


@dataclass
class Loss:
    type: List[str] = field(default_factory=lambda: ["mse_loss"])
    weights: List[float] = field(default_factory=lambda: [1.0])


@dataclass
class Metric:
    type: List[str] = field(default_factory=lambda: ["psnr"])


@dataclass
class Optimizer:
    type: str = "AdamW"
    lr: float = 0.0005
    momentum: Optional[float] = None
    weight_decay: float = 0.0


@dataclass
class Scheduler:
    type: str = "warmup_cosine"
    warmup_steps: int = 500
    max_steps: int = MISSING
    start_lr: float = 0.0
    eta_min: float = 0.0


@dataclass
class Trainer:
    start_epoch: Optional[int] = 0
    num_epochs: int = 100
    steps_per_epoch: int = 500
    grad_accumulation_steps: int = 1
    statslogger: str = "wandb"
    project_name: str = "nerf_editor"
    experiment_name: Optional[str] = None
    output_directory: str = "outputs"
    seed: Optional[int] = None
    save_checkpoint_every: Optional[int] = 50
    run_validation_every: Optional[int] = 50
    train_visualize_every: Optional[int] = 100


@dataclass
class Evaluator:
    project_name: str = "nerf_editor"
    experiment_directory: str = MISSING
    checkpoint_id: Optional[int] = None


@dataclass
class Predictor(Evaluator):
    resolution: int = 64
    point_subset: Optional[int] = None
    mcubes_threshold: Optional[float] = 0.5
    upsampling_steps: Optional[int] = 0
    mise_resolution: Optional[int] = None
    reconstruction_padding: Optional[float] = 0.0


@dataclass
class TrainConfigSchema:
    train_data: TrainData = TrainData()
    val_data: Optional[ValData] = ValData()
    model: Model = Model()
    renderer: Renderer = Renderer()
    loss: Loss = Loss()
    metric: Metric = Metric()
    optimizer: Optimizer = Optimizer()
    scheduler: Optional[Scheduler] = None
    train: Train = Train()
    validation: Optional[Validation] = Validation()
    trainer: Trainer = Trainer()


@dataclass
class InferConfigSchema:
    data: ValData = ValData()
    model: Model = Model()
    renderer: Renderer = Renderer()
    inference: Validation = Validation()
    predictor: Predictor = Predictor()
