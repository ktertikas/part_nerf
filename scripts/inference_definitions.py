from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from autodecoder_definitions import Model
from nerf_definitions import Predictor, Renderer, ValData


@dataclass
class Editor(Predictor):
    editing_ids: List[int] = field(default_factory=lambda: [0, 1])
    swapping_part_ids: List[int] = field(default_factory=lambda: [15, 14, 13, 11, 2])
    mixing_parts_first: List[int] = field(default_factory=lambda: [15, 14, 13, 11, 2])
    mixing_parts_second: List[int] = field(default_factory=lambda: [15, 14, 13, 11, 2])
    interpolation_part_id: int = 15
    removal_part_ids: List[int] = field(default_factory=lambda: [15, 11])
    addition_part_ids: List[int] = field(
        default_factory=lambda: [
            15,
        ]
    )
    transformation_part_id: int = 15
    near: float = 0.5
    far: float = 2.5
    height: int = 256
    width: int = 256
    num_samples: int = 64
    rays_chunk: int = 2048
    dataset_type: str = "shapenet"
    save_gifs: bool = False
    with_texture: bool = True


@dataclass
class EditConfigSchema:
    model: Model = Model()
    renderer: Renderer = Renderer()
    editor: Editor = Editor()


@dataclass
class Generator(Predictor):
    near: float = 0.5
    far: float = 2.5
    height: int = 256
    width: int = 256
    num_samples: int = 64
    rays_chunk: int = 2048
    num_generations: int = 10
    with_texture: bool = False
    with_parts: bool = False
    save_gifs: bool = False


@dataclass
class InferenceConfigSchema:
    model: Model = Model()
    renderer: Optional[Renderer] = None


@dataclass
class EvaluationConfigSchema:
    data: ValData = ValData()
    model: Model = Model()
    renderer: Renderer = Renderer()
