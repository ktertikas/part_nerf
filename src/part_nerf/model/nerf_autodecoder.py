from typing import Dict, Optional

import torch
from torch import nn

from .decomposition_network import get_decomposition_network
from .embedding_network import get_embedding_network
from .occupancy_network import build_occupancy_network
from .color_network import build_color_network
from .structure_network import build_structure_network
from .rays_associator import get_ray_associator
from .transformer import get_transformer_encoder


class NerfAutodecoder(nn.Module):
    def __init__(
        self,
        shape_embedding_network: nn.Module,
        texture_embedding_network: nn.Module,
        shape_decomposition_network: nn.Module,
        texture_decomposition_network: nn.Module,
        structure_network: nn.Module,
        occupancy_network: nn.Module,
        color_network: nn.Module,
        ray_part_associator: Optional[nn.Module] = None,
    ):
        """The main implementation of our Autodecoder method.

        Args:
            shape_embedding_network (nn.Module): The embedding network describing the shape.
            texture_embedding_network (nn.Module): The embedding network describing the texture.
            shape_decomposition_network (nn.Module): The decomposition network for the shape.
            texture_decomposition_network (nn.Module): The decomposition network for the texture.
            structure_network (nn.Module): The network predicting the primitive parameters for each part.
            occupancy_network (nn.Module): The network predicting the implicit field.
            color_network (nn.Module): The network predicting the color value per point.
            ray_part_associator (Optional[nn.Module], optional): The ray-part associator. Defaults to None.
        """
        super().__init__()
        self._shape_embedding_network = shape_embedding_network
        self._texture_embedding_network = texture_embedding_network
        self._shape_decomposition_network = shape_decomposition_network
        self._texture_decomposition_network = texture_decomposition_network
        self._structure_network = structure_network
        self._occupancy_network = occupancy_network
        self._color_network = color_network
        self._ray_part_associator = ray_part_associator

    def get_shape_embedding(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        shape_embedding = X.get("shape_embedding")
        if shape_embedding is not None:
            return shape_embedding
        shape_id = X.get("scene_id")
        assert shape_id is not None
        return self._shape_embedding_network(shape_id)

    def get_texture_embedding(
        self, X: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        texture_embedding = X.get("texture_embedding")
        if texture_embedding is not None:
            return texture_embedding
        texture_id = X.get("scene_id")
        assert texture_id is not None
        return self._texture_embedding_network(texture_id)

    def compute_part_shape_embeddings(
        self, pred_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        part_shape_embeddings = self._shape_decomposition_network(
            pred_dict["shape_embedding"]
        )
        return part_shape_embeddings

    def compute_part_texture_embeddings(
        self, pred_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        part_texture_embeddings = self._texture_decomposition_network(
            pred_dict["texture_embedding"]
        )
        return part_texture_embeddings

    def compute_part_params(
        self, pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pred_dict.update(self._structure_network(pred_dict))
        return pred_dict

    def compute_occupancy_field(
        self, X: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pred_dict.update(self._occupancy_network(X, pred_dict))
        return pred_dict

    def compute_color_field(
        self, X: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._ray_part_associator is not None:
            pred_dict.update(self._ray_part_associator(X, pred_dict))
        pred_dict.update(self._color_network(X, pred_dict))
        return pred_dict

    def forward(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # computing per-part shape and texture embeddings
        pred_dict = self.forward_part_features(X)
        # computing per-part parameters
        pred_dict.update(self.compute_part_params(pred_dict))
        # computing occupancy predictions
        pred_dict.update(self.compute_occupancy_field(X, pred_dict))
        # computing color predictions
        pred_dict.update(self.compute_color_field(X, pred_dict))
        return pred_dict

    # Utility functions for intermediate step computations,
    # especially used during editing operations
    def get_random_shape_embeddings(self, num_embeddings: int):
        shape_embeddings = self._shape_embedding_network.get_random_embeddings(
            num_embeddings
        )
        return shape_embeddings

    def get_random_texture_embeddings(self, num_embeddings: int):
        texture_embeddings = self._texture_embedding_network.get_random_embeddings(
            num_embeddings
        )
        return texture_embeddings

    def forward_part_features(
        self, X: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pred_dict = {}
        # computing per-part shape and texture embeddings
        pred_dict["shape_embedding"] = self.get_shape_embedding(X)
        pred_dict["texture_embedding"] = self.get_texture_embedding(X)
        pred_dict["part_shape_features"] = self.compute_part_shape_embeddings(pred_dict)
        pred_dict["part_texture_features"] = self.compute_part_texture_embeddings(
            pred_dict
        )
        return pred_dict

    def forward_part_features_and_params(
        self, X: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # computing per-part features
        pred_dict = self.forward_part_features(X)
        # computing per-part parameters
        pred_dict.update(self.compute_part_params(pred_dict))
        return pred_dict

    def forward_part_features_and_params_from_random(
        self, num_embeddings: int
    ) -> Dict[str, torch.Tensor]:
        X = {}
        X["shape_embedding"] = self.get_random_shape_embeddings(num_embeddings)
        X["texture_embedding"] = self.get_random_texture_embeddings(num_embeddings)
        pred_dict = self.forward_part_features_and_params(X)
        return pred_dict

    def forward_part_occupancies(
        self, X: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pred_dict = self.forward_part_features_and_params(X)
        # computing occupancy predictions
        pred_dict.update(self.compute_occupancy_field(X, pred_dict))
        return pred_dict

    def forward_occupancy_field_from_part_features(
        self, X: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # copy dictionary in order to generate new predictions in every method call
        predictions = pred_dict.copy()
        # computing per-part parameters
        predictions.update(self.compute_part_params(predictions))
        # computing occupancy predictions
        predictions.update(self.compute_occupancy_field(X, predictions))
        # Removing intermediate outputs that are only used in color prediction to free up memory
        predictions.pop("point_part_features", None)
        predictions.pop("points_transformed", None)
        return predictions

    def forward_color_field_from_part_features(
        self, X: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # copy dictionary in order to generate new predictions in every method call
        predictions = pred_dict.copy()
        # computing per-part parameters
        predictions.update(self.compute_part_params(predictions))
        # computing occupancy predictions
        predictions.update(self.compute_occupancy_field(X, predictions))
        # computing color field predictions
        predictions.update(self.compute_color_field(X, predictions))
        return predictions

    def forward_occupancy_field_from_part_preds(
        self, X: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # copy dictionary in order to generate new predictions in every method call
        predictions = pred_dict.copy()
        # computing occupancy predictions
        predictions.update(self.compute_occupancy_field(X, predictions))
        # Removing intermediate outputs that are only used in color prediction to free up memory
        predictions.pop("point_part_features", None)
        predictions.pop("points_transformed", None)
        return predictions

    def forward_color_field_from_part_preds(
        self, X: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # copy dictionary in order to generate new predictions in every method call
        predictions = pred_dict.copy()
        # computing occupancy predictions
        predictions.update(self.compute_occupancy_field(X, predictions))
        # computing color field predictions
        predictions.update(self.compute_color_field(X, predictions))
        return predictions


def build_nerf_autodecoder(config: Dict) -> NerfAutodecoder:
    # config parsing
    shape_embedding_config = config["shape_embedding_network"]
    texture_embedding_config = config["texture_embedding_network"]
    shape_decomposition_config = config["shape_decomposition_network"]
    shape_encoder_config = shape_decomposition_config["encoder"]
    texture_decomposition_config = config["texture_decomposition_network"]
    texture_encoder_config = texture_decomposition_config["encoder"]
    associator_config = config["ray_associator"]
    color_network_config = config["color_network"]
    occupancy_config = config.get("occupancy_network")
    # embedding networks
    shape_embedding_network = get_embedding_network(
        shape_embedding_config["type"], shape_embedding_config
    )
    texture_embedding_network = get_embedding_network(
        texture_embedding_config["type"], texture_embedding_config
    )
    # decomposition networks
    shape_encoder = None
    if shape_encoder_config is not None:
        shape_encoder = get_transformer_encoder(
            shape_encoder_config["type"], shape_encoder_config
        )
    shape_decomposition_network = get_decomposition_network(
        shape_decomposition_config["type"],
        shape_decomposition_config,
        encoder=shape_encoder,
    )
    texture_encoder = None
    if texture_encoder_config is not None:
        texture_encoder = get_transformer_encoder(
            texture_encoder_config["type"], texture_encoder_config
        )
    texture_decomposition_network = get_decomposition_network(
        texture_decomposition_config["type"],
        texture_decomposition_config,
        encoder=texture_encoder,
    )
    # structure network
    structure_network = build_structure_network(config)
    # color network
    color_network = build_color_network(color_network_config)
    # occupancy network
    occupancy_network = build_occupancy_network(occupancy_config)
    # ray_part associator
    ray_part_associator = None
    if associator_config is not None:
        ray_part_associator = get_ray_associator(associator_config)

    return NerfAutodecoder(
        shape_embedding_network=shape_embedding_network,
        texture_embedding_network=texture_embedding_network,
        shape_decomposition_network=shape_decomposition_network,
        texture_decomposition_network=texture_decomposition_network,
        structure_network=structure_network,
        occupancy_network=occupancy_network,
        color_network=color_network,
        ray_part_associator=ray_part_associator,
    )
