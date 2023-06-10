from typing import Dict, List

import torch
from torch import nn

from .rotations import get_rotation_instance
from .scale import get_scale_instance
from .shape import get_shape_instance
from .translations import get_translation_instance


class StructureNetwork(nn.Module):
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self._layers = nn.ModuleList(layers)

    def forward(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for l in self._layers:
            X = l(X)
        return X


def build_structure_network(model_config: Dict) -> StructureNetwork:
    layers = model_config["structure_network"].get("layers", [])
    return StructureNetwork(layers=get_layer_instances(model_config, layers))


def get_layer_instances(model_config: Dict, layers: List) -> List[nn.Module]:
    factories = {
        "translations": get_translation_instance,
        "rotations": get_rotation_instance,
        "scale": get_scale_instance,
        "shape": get_shape_instance,
    }
    layer_list = []
    for lname in layers:
        category, layer = lname.split(":")
        layer_list.append(factories[category](layer, model_config))
    return layer_list
