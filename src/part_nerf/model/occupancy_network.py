from functools import partial
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from ..primitive_utils import (
    apply_sigmoid_to_inside_outside_function,
    get_implicit_surface_from_inside_outside_function,
    inside_outside_function_ellipsoid,
    transform_to_primitives_centric_system,
)
from .occupancy_base import MiniOccupancyNet
from .positional_encoding import FixedPositionalEncoding


class MiniOccupancyFunction(nn.Module):
    def __init__(
        self,
        feature_size: int,
        in_dim: int = 3,
        out_dim: int = 1,
        n_blocks: int = 1,
        pos_encoding_size: int = None,
        sharpness_inside: float = 10.0,
        sharpness_outside: float = 10.0,
        norm_method: str = None,
        chunk_size: int = -1,
    ):
        super().__init__()
        self._sharpness_inside = sharpness_inside
        self._sharpness_outside = sharpness_outside
        input_dim = in_dim
        self._out_dim = out_dim
        if pos_encoding_size is not None:
            input_dim += in_dim * pos_encoding_size
            self.pos_encoding = FixedPositionalEncoding(pos_encoding_size)
        else:
            self.pos_encoding = nn.Identity()
        self.occupancy_model = MiniOccupancyNet(
            dim=input_dim,
            c_dim=feature_size,
            out_dim=out_dim,
            hidden_size=256,
            n_blocks=n_blocks,
            with_sigmoid=False,
            norm_method=norm_method,
        )
        self._chunk_size = chunk_size

    def calculate_occupancy(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # positional encoding
        pos_enc = self.pos_encoding(points)
        out = self.occupancy_model(pos_enc, features)
        occupancy = out[..., 0]
        point_features = None
        if self._out_dim > 1:
            point_features = out[..., 1:]
        return occupancy, point_features

    def get_occupancy_implicit_field(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, M, _ = points.shape
        F = []
        point_features_list = []
        for i in range(M):
            current_primitive_feature = features[:, i, :]
            current_samples = points[..., i, :]
            preds, point_features = self.calculate_occupancy(
                current_samples, current_primitive_feature
            )  # (B, N), (B, N, out_dim-1)
            F.append(preds)
            if point_features is not None:
                point_features_list.append(point_features)
        # Occupancy network fine implicit_field
        F = torch.stack(F, dim=-1)  # (B, N, M)
        assert F.shape == (B, N, M)
        implicit_values_pred = apply_sigmoid_to_inside_outside_function(
            F, self._sharpness_inside, self._sharpness_outside
        )
        # Point features if they exist
        point_features = None
        if len(point_features_list) > 0:
            point_features = torch.stack(point_features_list, dim=-1).transpose(
                -1, -2
            )  # (B, N, M, out_dim-1)
        return implicit_values_pred, point_features

    def forward(
        self, X: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        translations = pred_dict.get("translations")  # (B, M, 3)
        part_shape_features = pred_dict.get(
            "part_shape_features"
        )  # (B, M, feature_size + embedding_size)
        rotations = pred_dict.get("rotations", None)  # (B, M, 4)

        sampled_points = X["ray_points"]
        B, N, num_points, _ = sampled_points.shape
        sampled_points_reshaped = sampled_points.reshape((B, N * num_points, -1))
        if self._chunk_size == -1:
            points_split_list = [sampled_points_reshaped]
        else:
            points_split_list = torch.split(
                sampled_points_reshaped, self._chunk_size, dim=1
            )
        M = translations.shape[1]  # num_parts
        implicit_values_list = []
        transformed_points_list = []
        for points in points_split_list:
            num_split_points = points.shape[1]
            # Transform the 3D points from world-coordinates to primitive-centric
            # coordinates with size BxNxMx3
            points_transformed = transform_to_primitives_centric_system(
                points, translations, rotations
            )
            assert points_transformed.shape == (B, num_split_points, M, 3)
            transformed_points_list.append(points_transformed)

            implicit_values_pred, _ = self.get_occupancy_implicit_field(
                points_transformed, part_shape_features
            )
            implicit_values_list.append(implicit_values_pred)
        implicit_values = torch.cat(implicit_values_list, dim=1)
        assert implicit_values.shape == (B, N * num_points, M)
        transformed_points = torch.cat(transformed_points_list, dim=1)
        assert transformed_points.shape == (B, N * num_points, M, 3)
        pred_dict["implicit_field"] = implicit_values.reshape((B, N, num_points, M))
        pred_dict["points_transformed"] = transformed_points.reshape(
            (B, N, num_points, -1)
        )
        return pred_dict


class MiniOccupancyWithEllipsoidsMaskingFunction(MiniOccupancyFunction):
    def __init__(
        self,
        feature_size: int,
        in_dim: int = 3,
        out_dim: int = 1,
        n_blocks: int = 1,
        pos_encoding_size: int = None,
        sharpness_inside: float = 10,
        sharpness_outside: float = 10,
        norm_method: str = None,
        chunk_size: int = -1,
    ):
        super().__init__(
            feature_size,
            in_dim,
            out_dim,
            n_blocks,
            pos_encoding_size,
            sharpness_inside,
            sharpness_outside,
            norm_method,
            chunk_size,
        )

    def get_occupancy_implicit_field(
        self, points: torch.Tensor, features: torch.Tensor, points_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, M, _ = points.shape
        assert points_mask.shape == (B, N, M)
        # split along batch dimension to calculate only the points that are needed
        point_batch_split = torch.split(points, 1, dim=0)
        features_batch_split = torch.split(features, 1, dim=0)
        points_mask_split = torch.split(points_mask, 1, dim=0)
        F_batch_list = []
        point_feature_batch_list = []
        for point_b, features_b, points_mask_b in zip(
            point_batch_split, features_batch_split, points_mask_split
        ):
            # Parallelization by batching over the part dimension for occupancy calculation
            num_inside_points_per_ellipsoid = points_mask_b.sum(dim=1)  # (1, M)
            max_inside_points = num_inside_points_per_ellipsoid.max().item()
            if max_inside_points == 0:
                F_batch_list.append(torch.zeros((1, N, M), device=point_b.device))
                if self._out_dim > 1:
                    point_feature_batch_list.append(
                        torch.zeros((1, N, M, self._out_dim - 1), device=point_b.device)
                    )
                continue
            # Keep only the points that are inside the ellipsoid in a batched way
            masked_points = torch.zeros(
                (1, max_inside_points, M, 3), device=point_b.device
            )
            batched_mask = torch.zeros(
                (1, max_inside_points, M), device=point_b.device, dtype=torch.bool
            )
            for i in range(M):
                true_inds = points_mask_b[0, :, i].nonzero().squeeze()
                num_inds = true_inds.numel()
                if num_inds > 0:
                    batched_mask[0, range(num_inds), i] = points_mask_b[0, true_inds, i]
                    masked_points[0, range(num_inds), i, :] = point_b[
                        0, true_inds, i, :
                    ]
            # Calculate the occupancy for each ellipsoid by batching along the part dimension
            preds_batched, point_features_batched = self.calculate_occupancy(
                masked_points[0].transpose(0, 1), features_b[0]
            )  # (M, max_inside_points), (M, max_inside_points, out_dim - 1)
            preds_batched = preds_batched.transpose(0, 1)[None, ...]
            assert preds_batched.shape == (1, max_inside_points, M)
            point_features_batched = (
                None
                if point_features_batched is None
                else point_features_batched.transpose(0, 1)[None, ...]
            )

            # Now we need to return the occupancy values and the point features back to the original shape
            F = (
                torch.zeros((1, N, M), device=point_b.device) - 100
            )  # -100 to make sure sigmoid is 0
            point_features = (
                None
                if point_features_batched is None
                else torch.zeros((1, N, M, self._out_dim - 1), device=point_b.device)
            )
            for i in range(M):
                true_inds = points_mask_b[0, :, i].nonzero().squeeze()
                num_inds = true_inds.numel()
                if num_inds > 0:
                    F[0, true_inds, i] = preds_batched[0, range(num_inds), i]
                    if point_features_batched is not None:
                        point_features[0, true_inds, i, :] = point_features_batched[
                            0, range(num_inds), i, :
                        ]

            F_batch_list.append(F)
            if point_features is not None:
                point_feature_batch_list.append(point_features)

        # Occupancy network fine implicit_field
        F = torch.cat(F_batch_list, dim=0)  # (B, N, M)
        assert F.shape == (B, N, M)
        # TODO: Check if there is an issue here
        implicit_values_pred = apply_sigmoid_to_inside_outside_function(
            F, self._sharpness_inside, self._sharpness_outside
        )
        # Point features if they exist
        point_features = None
        if len(point_feature_batch_list) > 0:
            point_features = torch.cat(
                point_feature_batch_list, dim=0
            )  # (B, N, M, out_dim-1)
        return implicit_values_pred, point_features

    def forward(
        self, X: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        translations = pred_dict.get("translations")  # (B, M, 3)
        part_shape_features = pred_dict.get(
            "part_shape_features"
        )  # (B, M, feature_size + embedding_size)
        rotations = pred_dict.get("rotations", None)  # (B, M, 4)
        alphas = pred_dict.get("scale")  # (B, M, 3)

        sampled_points = X["ray_points"]
        B, N, num_points, _ = sampled_points.shape
        sampled_points_reshaped = sampled_points.reshape((B, N * num_points, -1))
        if self._chunk_size == -1:
            points_split_list = [sampled_points_reshaped]
        else:
            points_split_list = torch.split(
                sampled_points_reshaped, self._chunk_size, dim=1
            )
        M = translations.shape[1]  # num_parts
        ellipsoid_inside_outside_list = []
        transformed_points_list = []
        for points in points_split_list:
            num_split_points = points.shape[1]
            # Transform the 3D points from world-coordinates to primitive-centric
            # coordinates with size BxNxMx3
            points_transformed = transform_to_primitives_centric_system(
                points, translations, rotations
            )
            assert points_transformed.shape == (B, num_split_points, M, 3)
            transformed_points_list.append(points_transformed)

            # Ellipsoid inside-outside function
            ellipsoid_inside_outside_F = inside_outside_function_ellipsoid(
                points_transformed, alphas
            )
            ellipsoid_inside_outside_list.append(ellipsoid_inside_outside_F)
        inside_outside_values = torch.cat(ellipsoid_inside_outside_list, dim=1)
        coarse_implicit_values = get_implicit_surface_from_inside_outside_function(
            inside_outside_values,
            sharpness_inside=self._sharpness_inside,
            sharpness_outside=self._sharpness_outside,
        )
        assert inside_outside_values.shape == (B, N * num_points, M)
        inside_outside_values = inside_outside_values.reshape((B, N, num_points, M))
        coarse_implicit_values = coarse_implicit_values.reshape((B, N, num_points, M))
        transformed_points = torch.cat(transformed_points_list, dim=1)
        assert transformed_points.shape == (B, N * num_points, M, 3)

        # Create mask for ray points that are inside an ellipsoid
        points_inside_ellipsoid = inside_outside_values <= 1  # (B, N, num_points, M)

        implicit_values, point_features = self.get_occupancy_implicit_field(
            transformed_points,
            part_shape_features,
            points_inside_ellipsoid.reshape((B, N * num_points, M)),
        )
        assert implicit_values.shape == (B, N * num_points, M)
        implicit_values = implicit_values.reshape((B, N, num_points, M))
        # filter out the implicit values that are outside the ellipsoids
        implicit_values = coarse_implicit_values * implicit_values

        pred_dict["implicit_field"] = implicit_values
        pred_dict["coarse_implicit_field"] = coarse_implicit_values
        pred_dict["points_transformed"] = transformed_points.reshape(
            (B, N, num_points, M, -1)
        )
        pred_dict["points_mask"] = points_inside_ellipsoid
        if point_features is not None:
            pred_dict["point_part_features"] = point_features.reshape(
                (B, N, num_points, M, -1)
            )
        return pred_dict


class MultiOccupancyWithEllipsoidsMaskingFunction(
    MiniOccupancyWithEllipsoidsMaskingFunction
):
    def __init__(
        self,
        feature_size: int,
        num_parts: int,
        in_dim: int = 3,
        out_dim: int = 1,
        n_blocks: int = 1,
        pos_encoding_size: int = None,
        sharpness_inside: float = 10,
        sharpness_outside: float = 10,
        norm_method: str = None,
        chunk_size: int = -1,
    ):
        super().__init__(
            feature_size,
            in_dim,
            out_dim,
            n_blocks,
            pos_encoding_size,
            sharpness_inside,
            sharpness_outside,
            norm_method,
            chunk_size,
        )
        self.occupancy_model = nn.ModuleList(
            [
                MiniOccupancyNet(
                    dim=in_dim,
                    c_dim=feature_size,
                    out_dim=out_dim,
                    hidden_size=256,
                    n_blocks=n_blocks,
                    with_sigmoid=False,
                    norm_method=norm_method,
                )
                for _ in range(num_parts)
            ]
        )

    def calculate_occupancy(
        self, points: torch.Tensor, features: torch.Tensor, primitive_idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # positional encoding
        pos_enc = self.pos_encoding(points)
        out = self.occupancy_model[primitive_idx](pos_enc, features)
        occupancy = out[..., 0]
        point_features = None
        if self._out_dim > 1:
            point_features = out[..., 1:]
        return occupancy, point_features

    def get_occupancy_implicit_field(
        self, points: torch.Tensor, features: torch.Tensor, points_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, M, _ = points.shape
        F = []
        point_features_list = []
        for i in range(M):
            current_primitive_feature = features[:, i, :]
            current_samples = points[..., i, :]
            preds, point_features = self.calculate_occupancy(
                current_samples, current_primitive_feature, primitive_idx=i
            )  # (B, N), (B, N, out_dim-1)
            F.append(preds)
            if point_features is not None:
                point_features_list.append(point_features)
        # Occupancy network fine implicit_field
        F = torch.stack(F, dim=-1)  # (B, N, M)
        assert F.shape == (B, N, M)
        implicit_values_pred = apply_sigmoid_to_inside_outside_function(
            F, self._sharpness_inside, self._sharpness_outside
        )
        # Point features if they exist
        point_features = None
        if len(point_features_list) > 0:
            point_features = torch.stack(point_features_list, dim=-1).transpose(
                -1, -2
            )  # (B, N, M, out_dim-1)
        return implicit_values_pred, point_features


def build_occupancy_network(config: Dict) -> nn.Module:
    model_type = config["type"]
    sharpness_inside = config.get("sharpness_inside", 10.0)
    sharpness_outside = config.get("sharpness_outside", 10.0)
    normalization_method = config.get("normalization_method", None)
    num_blocks = config.get("num_blocks", 1)
    positional_encoding_size = config.get("positional_encoding_size", None)
    chunk_size = config.get("chunk_size", 10000)
    embedding_size = config["embedding_size"]
    num_parts = config.get("num_parts")
    out_dim = config.get("output_dim", 1)
    implicit_factory = {
        "masked_occ": partial(
            MiniOccupancyWithEllipsoidsMaskingFunction,
            feature_size=embedding_size,
            pos_encoding_size=positional_encoding_size,
            out_dim=out_dim,
            n_blocks=num_blocks,
            sharpness_inside=sharpness_inside,
            sharpness_outside=sharpness_outside,
            norm_method=normalization_method,
            chunk_size=chunk_size,
        ),
        "multi_masked_occ": partial(
            MultiOccupancyWithEllipsoidsMaskingFunction,
            feature_size=embedding_size,
            num_parts=num_parts,
            pos_encoding_size=positional_encoding_size,
            out_dim=out_dim,
            n_blocks=num_blocks,
            sharpness_inside=sharpness_inside,
            sharpness_outside=sharpness_outside,
            norm_method=normalization_method,
            chunk_size=chunk_size,
        ),
    }
    return implicit_factory[model_type]()
