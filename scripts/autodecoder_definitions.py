"""Script that defines the input argument types coming from the configuration files."""
from dataclasses import dataclass, field
from typing import List, Optional

from nerf_definitions import InferConfigSchema as InferConfigSchemaBase
from nerf_definitions import Metric, Optimizer, Predictor, Renderer, Train
from nerf_definitions import TrainConfigSchema as TrainConfigSchemaBase
from nerf_definitions import TrainData
from omegaconf import MISSING


@dataclass
class EmbeddingNetwork:
    type: str = "simple"
    embedding_size: int = 128
    num_embeddings: int = 1  # equal to the number of instances in the dataset
    max_norm: Optional[float] = None


@dataclass
class TransformerEncoder:
    type: str = "simple"
    input_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop: float = 0.0
    attn_drop: float = 0.0
    activation: str = "relu"
    normalization: Optional[str] = None


@dataclass
class DecompositionNetwork:
    type: str = "simple"
    num_parts: int = 16
    embedding_size: int = 128
    output_size: int = 128
    encoder: Optional[TransformerEncoder] = TransformerEncoder()


@dataclass
class StructureNetwork:
    layers: List = field(
        default_factory=lambda: [
            "translations:embedding",
            "rotations:embedding",
            "scale:embedding",
        ]
    )
    scale_min_a: float = 0.005
    scale_max_a: float = 0.5


@dataclass
class RayAssociator:
    type: str = "occupancy"
    implicit_threshold: float = 0.5


@dataclass
class ColorEncoder:
    type: str = "residual"
    input_dims: Optional[int] = None
    out_dims: int = 3
    proj_dims: List[List[int]] = field(
        default_factory=lambda: [[450, 256, 256], [256, 256, 256], [256, 128, 64]]
    )
    activation: str = "relu"
    last_activation: Optional[str] = "sigmoid"


@dataclass
class ColorNetwork:
    type: str = "hard"
    encoder: ColorEncoder = ColorEncoder()
    pts_proj_dims: int = 20
    dir_proj_dims: Optional[int] = None
    dir_coord_system: str = "global"


@dataclass
class OccupancyNetwork:
    type: str = "simple_occ"
    embedding_size: int = 256
    num_parts: Optional[int] = None
    output_dim: int = 1
    num_blocks: int = 1
    sharpness_inside: float = 10.0
    sharpness_outside: float = 10.0
    normalization_method: Optional[str] = None
    positional_encoding_size: Optional[int] = None
    chunk_size: int = 10000


@dataclass
class Model:
    coarse_fine: bool = False
    shape_embedding_network: EmbeddingNetwork = EmbeddingNetwork()
    texture_embedding_network: EmbeddingNetwork = EmbeddingNetwork()
    shape_decomposition_network: DecompositionNetwork = DecompositionNetwork()
    texture_decomposition_network: DecompositionNetwork = DecompositionNetwork()
    structure_network: StructureNetwork = StructureNetwork()
    occupancy_network: OccupancyNetwork = OccupancyNetwork()
    color_network: ColorNetwork = ColorNetwork()
    ray_associator: Optional[RayAssociator] = RayAssociator()


@dataclass
class Loss:
    type: List[str] = field(default_factory=lambda: ["mse_loss"])
    weights: List[float] = field(default_factory=lambda: [1.0])
    num_inside_rays: int = 10


@dataclass
class TrainConfigSchema(TrainConfigSchemaBase):
    model: Model = Model()
    loss: Loss = Loss()


@dataclass
class InferConfigSchema(InferConfigSchemaBase):
    model: Model = Model()


@dataclass
class Invertor(Predictor):
    num_iters: int = 250
    with_texture: bool = False
    grad_accumulation_steps: int = 1


@dataclass
class InversionConfigSchema:
    data: TrainData = TrainData()
    model: Model = Model()
    renderer: Renderer = Renderer()
    loss: Loss = Loss()
    optimizer: Optimizer = Optimizer()
    metric: Metric = Metric()
    invertor: Invertor = Invertor()
    invert: Train = Train()
