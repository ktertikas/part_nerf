from functools import partial
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from ..primitive_utils import (
    transform_to_primitives_centric_system,
    transform_unit_directions_to_primitives_centric_system,
)
from .mlp_encoder import MLPEncoder, ResidualEncoder
from .positional_encoding import FixedPositionalEncoding
from .utils import activation_factory


class ColorNetworkSoftAssignment(nn.Module):
    def __init__(
        self,
        color_encoder: MLPEncoder,
        pts_proj_dims: int = 20,
        dir_proj_dims: Optional[int] = None,
        dir_coord_system: str = "global",
    ):
        super().__init__()
        # Positional Encoding for the input 3D points and view directions
        self.pts_proj_dims = pts_proj_dims
        self.pe_pts = FixedPositionalEncoding(proj_dims=pts_proj_dims)
        self.dir_proj_dims = dir_proj_dims if dir_proj_dims else 0
        self.pe_dir = (
            FixedPositionalEncoding(proj_dims=dir_proj_dims)
            if dir_proj_dims
            else nn.Identity()
        )

        # Sub-network used for predicting the view-dependent RGB color
        self.color_encoder = color_encoder
        assert dir_coord_system in ["global", "primitive"]
        self._dir_coord_system = dir_coord_system

    def get_ray_colors_per_primitive(
        self,
        points_f: torch.Tensor,
        directions_f: torch.Tensor,
        primitive_features: torch.Tensor,
    ) -> torch.Tensor:
        B, N, M, num_parts, _ = points_f.shape
        ray_colors_list = []
        for i in range(num_parts):
            current_samples = points_f[
                ..., i, :
            ]  # (B, N, M, self.pts_proj_dims * 3 + 3)
            current_primitive_features = primitive_features[
                ..., i, :
            ]  # (B, N, M, feature_size)
            input_features = torch.cat(
                [
                    current_samples,
                    current_primitive_features,
                    directions_f,
                ],
                dim=-1,
            )  # (B, N, M, self.pts_proj_dims * 3 + self.dir_proj_dims * 3 + feature_size + 6)

            ray_colors_pred = self.color_encoder(input_features)  # (B, N, M, 3)
            ray_colors_list.append(ray_colors_pred)

        ray_colors_per_primitive = torch.stack(ray_colors_list, dim=-1).reshape(
            (B, N, M, num_parts, -1)
        )
        return ray_colors_per_primitive

    def forward(
        self, X: Dict[str, Any], prediction_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        ray_points = X["ray_points"]
        ray_directions = X["ray_directions"]
        # Extract the batch size, the number of rays and the number of points
        # into some local variables
        B, N, M, _ = ray_points.shape
        assert ray_directions.shape == (B, N, 3)
        # Normalize the ray_directions to unit l2 norm.
        ray_directions_normed = nn.functional.normalize(ray_directions, dim=-1)

        # Get translation and rotation of primitives
        translations = prediction_dict.get("translations")  # (B, num_parts, 3)
        rotations = prediction_dict.get("rotations")  # (B, num_parts, 4)

        num_parts = translations.shape[1]
        # Get primitive features
        primitive_features = prediction_dict.pop(
            "point_part_features"
        )  # (B, N, M, num_parts, feature_size)
        texture_features = prediction_dict.get(
            "part_texture_features"
        )  # (B, num_parts, texture_feature)
        if texture_features is not None:
            # use texture feature for color prediction
            primitive_features = torch.cat(
                [
                    primitive_features,
                    texture_features[:, None, None, ...].expand(-1, N, M, -1, -1),
                ],
                dim=-1,
            )  # (B, N, M, num_parts, feature_size + texture_feature)

        ray_points_transformed = prediction_dict.pop("points_transformed", None)
        if ray_points_transformed is None:
            # Transform to the local coordinate system of each primitive
            ray_points_transformed = transform_to_primitives_centric_system(
                ray_points.view(B, N * M, -1),
                translations=translations,
                rotation_angles=rotations,
            ).reshape((B, N, M, num_parts, -1))

        if self._dir_coord_system == "primitive":
            # Turn ray directions to primitive coordinate system
            ray_directions_transformed = (
                transform_unit_directions_to_primitives_centric_system(
                    ray_directions_normed, rotation_angles=rotations
                )
            )
            ray_directions_normed = nn.functional.normalize(
                ray_directions_transformed, dim=-1
            )

        # Apply the positional encoding along each dimension of the ray_points
        pts_f = self.pe_pts(ray_points_transformed)
        assert pts_f.shape == (B, N, M, num_parts, self.pts_proj_dims * 3 + 3)
        dir_f = self.pe_dir(ray_directions_normed)
        assert dir_f.shape == (B, N, self.dir_proj_dims * 3 + 3)
        # Expand the ray_directions_f in order to to the concatentation with
        # the feature vector
        dir_f = dir_f[:, :, None, :].expand(-1, -1, M, -1)

        ray_colors = self.get_ray_colors_per_primitive(pts_f, dir_f, primitive_features)
        prediction_dict["ray_colors"] = ray_colors

        return prediction_dict


class ColorNetworkHardAssignment(nn.Module):
    """This class implements the Mini-NeRF model, that is basically only a color predictor.

    Args:
        color_encoder (MLPEncoder): An MLP that computes the view-dependent RGB
            color for each ray.
        pts_proj_dims (int): The positional encoding bands per point dimension.
            Defaults to 20.
        dir_proj_dims (Optional[int]): The positional encoding bands per direction vector dimension.
            Defaults to None.
        dir_coord_system (str): The coordinate system in which the direction vectors live. Defaults to "global".
    """

    def __init__(
        self,
        color_encoder: MLPEncoder,
        pts_proj_dims: int = 20,
        dir_proj_dims: Optional[int] = None,
        dir_coord_system: str = "global",
    ):
        super().__init__()
        # Positional Encoding for the input 3D points and view directions
        self.pts_proj_dims = pts_proj_dims
        self.pe_pts = FixedPositionalEncoding(proj_dims=pts_proj_dims)
        self.dir_proj_dims = dir_proj_dims if dir_proj_dims else 0
        self.pe_dir = (
            FixedPositionalEncoding(proj_dims=dir_proj_dims)
            if dir_proj_dims
            else nn.Identity()
        )

        # Sub-network used for predicting the view-dependent RGB color
        self.color_encoder = color_encoder
        assert dir_coord_system in ["global", "primitive"]
        self._dir_coord_system = dir_coord_system

    def get_ray_colors(
        self,
        points_f: torch.Tensor,
        directions_f: torch.Tensor,
        primitive_features: torch.Tensor,
    ) -> torch.Tensor:
        input_features = torch.cat(
            [
                points_f,
                primitive_features,
                directions_f,
            ],
            dim=-1,
        )  # (B, N, M, self.pts_proj_dims * 3 + self.dir_proj_dims * 3 + feature_size + 6)
        ray_colors_pred = self.color_encoder(input_features)  # (B, N, M, 3)
        return ray_colors_pred

    def forward(
        self, X: Dict[str, Any], prediction_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        ray_points = X["ray_points"]
        ray_directions = X["ray_directions"]
        # Extract the batch size, the number of rays and the number of points
        # into some local variables
        B, N, M, _ = ray_points.shape
        assert ray_directions.shape == (B, N, 3)
        # Normalize the ray_directions to unit l2 norm.
        ray_directions_normed = nn.functional.normalize(ray_directions, dim=-1)

        # Get translation and rotation of primitives
        translations = prediction_dict.get("translations")  # (B, num_parts, 3)
        rotations = prediction_dict.get("rotations")  # (B, num_parts, 4)
        rays_associations = prediction_dict.get("primitive_associations")  # (B, N)
        no_rendering_rays = prediction_dict.get("no_rendering")  # (B, N)

        num_parts = translations.shape[1]

        # Get primitive features
        primitive_features = prediction_dict.pop(
            "point_part_features"
        )  # (B, N, M, num_parts, feature_size)
        texture_features = prediction_dict.get(
            "part_texture_features"
        )  # (B, num_parts, texture_feature)
        if texture_features is not None:
            # use texture feature for color prediction
            primitive_features = torch.cat(
                [
                    primitive_features,
                    texture_features[:, None, None, ...].expand(-1, N, M, -1, -1),
                ],
                dim=-1,
            )  # (B, N, M, num_parts, feature_size + texture_feature)

        feature_size = primitive_features.shape[-1]

        ray_points_transformed = prediction_dict.pop("points_transformed", None)
        if ray_points_transformed is None:
            # Transform to the local coordinate system of each primitive
            ray_points_transformed = transform_to_primitives_centric_system(
                ray_points.view(B, N * M, -1),
                translations=translations,
                rotation_angles=rotations,
            ).reshape((B, N, M, num_parts, -1))

        if self._dir_coord_system == "primitive":
            # Turn ray directions to primitive coordinate system
            ray_directions_transformed = (
                transform_unit_directions_to_primitives_centric_system(
                    ray_directions_normed, rotation_angles=rotations
                )
            )
            ray_directions_normed = nn.functional.normalize(
                ray_directions_transformed, dim=-1
            )
            ray_directions_selected = torch.gather(
                ray_directions_normed,
                dim=2,
                index=rays_associations[..., None, None].expand(-1, -1, -1, 3),
            ).view(
                B, N, -1
            )  # (B, N, 3)
        else:
            ray_directions_selected = ray_directions_normed

        # Index rays, ray points etc by associations
        ray_points_selected = torch.gather(
            ray_points_transformed,
            dim=3,
            index=rays_associations[..., None, None, None].expand(-1, -1, M, -1, 3),
        ).view(
            B, N, M, -1
        )  # (B, N, M, 3)
        # before: (B, num_parts,feature_size), (B, N)
        # now: (B, N, M, num_parts, feature_size)
        primitive_features_selected = torch.gather(
            primitive_features,
            dim=3,
            index=rays_associations[..., None, None, None].expand(
                -1, -1, M, -1, feature_size
            ),
        ).view(
            B, N, M, -1
        )  # (B, N, M, feature_size)

        # Apply the positional encoding along each dimension of the ray_points
        pts_f = self.pe_pts(ray_points_selected)
        assert pts_f.shape == (B, N, M, self.pts_proj_dims * 3 + 3)
        dir_f = self.pe_dir(ray_directions_selected)
        assert dir_f.shape == (B, N, self.dir_proj_dims * 3 + 3)
        # Expand the ray_directions_f in order to to the concatentation with
        # the feature vector
        dir_f = dir_f[:, :, None, :].expand(-1, -1, M, -1)

        ray_colors = self.get_ray_colors(pts_f, dir_f, primitive_features_selected)
        prediction_dict["ray_colors"] = ray_colors
        return prediction_dict


def build_color_network(model_config) -> nn.Module:
    color_encoder = build_color_encoder(model_config["encoder"])
    network_factory = {
        "soft": partial(
            ColorNetworkSoftAssignment,
            color_encoder=color_encoder,
            pts_proj_dims=model_config.get("pts_proj_dims", 20),
            dir_proj_dims=model_config.get("dir_proj_dims", None),
            dir_coord_system=model_config.get("dir_coord_system", "global"),
        ),
        "hard": partial(
            ColorNetworkHardAssignment,
            color_encoder=color_encoder,
            pts_proj_dims=model_config.get("pts_proj_dims", 20),
            dir_proj_dims=model_config.get("dir_proj_dims", None),
            dir_coord_system=model_config.get("dir_coord_system", "global"),
        ),
    }
    network_type = model_config.get("type")
    return network_factory[network_type]()


def build_color_encoder(config: Dict) -> Union[MLPEncoder, ResidualEncoder]:
    encoder_type = config["type"]
    input_dims = config.get("input_dims", None)
    proj_dims = config.get(
        "proj_dims", [[450, 256, 256], [256, 256, 256], [256, 128, 64]]
    )
    out_dims = config.get("out_dims", 3)
    activation_name = config.get("activation", "relu")
    activation = (
        activation_factory(activation_name)() if activation_name else nn.Identity()
    )
    last_activation_name = config.get("last_activation", "sigmoid")
    last_activation = (
        activation_factory(last_activation_name)()
        if last_activation_name
        else nn.Identity()
    )

    if encoder_type == "mlp":
        encoder = MLPEncoder(
            input_dims=input_dims,
            proj_dims=proj_dims,
            non_linearity=activation,
            last_activation=last_activation,
        )
    elif encoder_type == "residual":
        encoder = ResidualEncoder(
            proj_dims=proj_dims,
            out_dims=out_dims,
            non_linearity=activation,
            last_activation=last_activation,
        )
    else:
        raise NotImplementedError(
            f"{encoder_type} does not belong to one of possible color encoders"
        )
    return encoder
