"""Utility functions for primitives, mostly borrowed from https://github.com/paschalidoud/hierarchical_primitives."""
from typing import Optional, Tuple

import numpy as np
import torch


def inside_outside_function_from_world_centric_points_sq(
    X: torch.Tensor,
    translations: torch.Tensor,
    rotations: torch.Tensor,
    alphas: torch.Tensor,
    epsilons: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Implicit Surface function calculation of each primitive M.

    Args:
        X (torch.Tensor): Tensor with size BxNx3, containing the 3D points, for which we want
            to compute the implicit_surface_function. B is the batch size and N
            is the number of points. The points are in world coordinates.
        translations (torch.Tensor): Tensor with size BxMx3, containing the translation
            vectors for the M primitives.
        rotations (torch.Tensor): Tensor with size BxMx4 containing the 4 quaternion
            values for the M primitives.
        alphas (torch.Tensor): Tensor with size BxMx3 containing the size along each axis for
            the M primitives.
        epsilons (torch.Tensor): Tensor with size BxMx2 containing the shape parameter along
            each axis for the M primitives.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the Tensor with size BxNxM
            of the implicit surface function of each primitive for the N points and the
            Tensor with size BxNxMx3 of the N points transformed in the M primitive
            centric coordinate systems.
    """
    B, N, _ = X.shape
    _, M, _ = translations.shape

    # Transform the 3D points from world-coordinates to primitive-centric
    # coordinates with size BxNxMx3
    X_transformed = transform_to_primitives_centric_system(X, translations, rotations)
    assert X_transformed.shape == (B, N, M, 3)

    # Compute the inside outside for every primitive
    F = inside_outside_function_sq(X_transformed, alphas, epsilons)
    return F, X_transformed


def transform_to_primitives_centric_system(
    X: torch.Tensor,
    translations: torch.Tensor,
    rotation_angles: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        X (torch.Tensor): Tensor with size BxNx3, containing the 3D points, where
            B is the batch size and N is the number of points.
        translations (torch.Tensor): Tensor with size BxMx3, containing the translation
            vectors for the M primitives.
        rotation_angles (Optional[torch.Tensor], optional): Tensor with size BxMx4 containing the 4 quaternion
            values for the M primitives. Defaults to None.

    Returns:
        torch.Tensor: Tensor with size BxNxMx3 containing the N points transformed in the
            M primitive centric coordinate systems.
    """
    # Make sure that all tensors have the right shape
    assert X.shape[0] == translations.shape[0]
    assert X.shape[-1] == 3
    assert translations.shape[-1] == 3
    if rotation_angles is not None:
        assert translations.shape[0] == rotation_angles.shape[0]
        assert translations.shape[1] == rotation_angles.shape[1]
        assert rotation_angles.shape[-1] == 4

    # Subtract the translation and get X_transformed with size BxNxMx3
    X_transformed = X.unsqueeze(2) - translations.unsqueeze(1)

    if rotation_angles is not None:
        R = quaternions_to_rotation_matrices(rotation_angles.view(-1, 4)).view(
            rotation_angles.shape[0], rotation_angles.shape[1], 3, 3
        )

        # Let us denote a point x_p in the primitive-centric coordinate system and
        # its corresponding point in the world coordinate system x_w. We denote the
        # transformation from the point in the world coordinate system to a point
        # in the primitive-centric coordinate system as x_p = R(x_w - t)
        X_transformed = R.unsqueeze(1).matmul(X_transformed.unsqueeze(-1))

    X_signs = (X_transformed > 0).float() * 2 - 1
    X_abs = X_transformed.abs()
    X_transformed = X_signs * torch.max(X_abs, X_abs.new_tensor(1e-5))

    return X_transformed.squeeze(-1)


def transform_unit_directions_to_primitives_centric_system(
    X: torch.Tensor, rotation_angles: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Args:
        X (torch.Tensor): Tensor with size BxNx3, containing the 3D points, where
            B is the batch size and N is the number of points.
        rotation_angles (Optional[torch.Tensor], optional): Tensor with size BxMx4 containing the 4 quaternion
            values for the M primitives. Defaults to None.
    Returns:
        torch.Tensor: Tensor with size BxNxMx3 containing the N points transformed in the
            M primitive centric coordinate systems.
    """
    # Make sure that all tensors have the right shape
    assert X.shape[-1] == 3
    assert rotation_angles.shape[-1] == 4
    M = rotation_angles.shape[1]

    # Subtract the translation and get X_transformed with size BxNxMx3
    X_transformed = X.unsqueeze(2).expand(-1, -1, M, -1)

    R = quaternions_to_rotation_matrices(rotation_angles.view(-1, 4)).view(
        rotation_angles.shape[0], rotation_angles.shape[1], 3, 3
    )

    X_transformed = R.unsqueeze(1).matmul(X_transformed.unsqueeze(-1))

    X_signs = (X_transformed > 0).float() * 2 - 1
    X_abs = X_transformed.abs()
    X_transformed = X_signs * torch.max(X_abs, X_abs.new_tensor(1e-5))

    return X_transformed.squeeze(-1)


def quaternions_to_rotation_matrices(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Args:
        quaternions (torch.Tensor): Tensor with size Kx4, where K is the number of quaternions
            we want to transform to rotation matrices.

    Returns:
        torch.Tensor: Tensor with size Kx3x3, that contains the computed rotation matrices.
    """
    K = quaternions.shape[0]
    # Allocate memory for a Tensor of size Kx3x3 that will hold the rotation
    # matrix along the x-axis
    R = quaternions.new_zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[:, 1] ** 2
    yy = quaternions[:, 2] ** 2
    zz = quaternions[:, 3] ** 2
    ww = quaternions[:, 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternions.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * quaternions[:, 1] * quaternions[:, 2]
    xz = s[:, 0] * quaternions[:, 1] * quaternions[:, 3]
    yz = s[:, 0] * quaternions[:, 2] * quaternions[:, 3]
    xw = s[:, 0] * quaternions[:, 1] * quaternions[:, 0]
    yw = s[:, 0] * quaternions[:, 2] * quaternions[:, 0]
    zw = s[:, 0] * quaternions[:, 3] * quaternions[:, 0]

    xx = s[:, 0] * xx
    yy = s[:, 0] * yy
    zz = s[:, 0] * zz

    idxs = torch.arange(K).to(quaternions.device)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R


def quaternions_to_rotation_matrices_np(quaternions: np.ndarray) -> np.ndarray:
    quaternions_torch = torch.from_numpy(quaternions)
    rotations_torch = quaternions_to_rotation_matrices(quaternions_torch)
    return rotations_torch.numpy()


def inside_outside_function_ellipsoid(
    X: torch.Tensor, shape_params: torch.Tensor
) -> torch.Tensor:
    """Calculation of the inside-outside function for the ellipsoid formulation.

    Args:
        X (torch.Tensor): Tensor with size BxNxMx3, containing the 3D points, where B is
            the batch size and N is the number of points.
        shape_params (torch.Tensor): Tensor with size BxMx3, containing the shape along each
            axis for the M primitives.

    Returns:
        torch.Tensor: Tensor with size BxNxM, containing the values of the inside-outside function.
    """
    B = X.shape[0]  # batch_size
    N = X.shape[1]  # number of points on target object
    M = X.shape[2]  # number of primitives

    # Make sure that both tensors have the right shape
    assert shape_params.shape[0] == B  # batch size
    assert shape_params.shape[1] == M  # number of primitives
    assert shape_params.shape[-1] == 3  # number of shape parameters
    assert X.shape[-1] == 3  # 3D points

    # Declare some variables
    a1 = shape_params[..., 0].unsqueeze(1)  # size Bx1xM
    a2 = shape_params[..., 1].unsqueeze(1)  # size Bx1xM
    a3 = shape_params[..., 2].unsqueeze(1)  # size Bx1xM

    # Add a small constant to points that are completely dead center to avoid
    # numerical issues in computing the gradient
    # zeros = X == 0
    # X[zeros] = X[zeros] + 1e-6
    X = ((X > 0).float() * 2 - 1) * torch.max(torch.abs(X), X.new_tensor(1e-6))
    F = (X[..., 0] / a1) ** 2 + (X[..., 1] / a2) ** 2 + (X[..., 2] / a3) ** 2

    # Sanity check to make sure that we have the expected size
    assert F.shape == (B, N, M)
    return F


def ellipsoid_volume(shape_params: torch.Tensor) -> torch.Tensor:
    """Calculation of the volume size of the ellipsoids.

    Args:
        shape_params (torch.Tensor): Tensor with size BxMx3, containing the shape along each
            axis for the M primitives.

    Returns:
        torch.Tensor: Tensor with size BxM, containing the volume for each single primitive.
    """
    B, M, _ = shape_params.shape

    # Declare some variables
    a1 = shape_params[..., 0]  # size BxM
    a2 = shape_params[..., 1]  # size BxM
    a3 = shape_params[..., 2]  # size BxM
    volume = 4.0 * np.pi * a1 * a2 * a3 / 3.0  # (B, M)
    return volume


def inside_outside_function_sq(
    X: torch.Tensor, shape_params: torch.Tensor, epsilons: torch.Tensor
) -> torch.Tensor:
    """Calculation of the inside-outside function for the superquadric formulation.

    Args:
        X (torch.Tensor): Tensor with size BxNxMx3, containing the 3D points, where B is
            the batch size and N is the number of points.
        shape_params (torch.Tensor): Tensor with size BxMx3, containing the shape along each
            axis for the M primitives.
        epsilons (torch.Tensor): Tensor with size BxMx2, containing the shape along the
            longitude and the latitude for the M primitives.

    Returns:
        torch.Tensor: Tensor with size BxNxM, containing the values of the inside-outside function.
    """
    B = X.shape[0]  # batch_size
    N = X.shape[1]  # number of points on target object
    M = X.shape[2]  # number of primitives

    # Make sure that both tensors have the right shape
    assert shape_params.shape[0] == B  # batch size
    assert epsilons.shape[0] == B  # batch size
    assert shape_params.shape[1] == M  # number of primitives
    assert shape_params.shape[1] == epsilons.shape[1]
    assert shape_params.shape[-1] == 3  # number of shape parameters
    assert epsilons.shape[-1] == 2  # number of shape parameters
    assert X.shape[-1] == 3  # 3D points

    # Declare some variables
    a1 = shape_params[..., 0].unsqueeze(1)  # size Bx1xM
    a2 = shape_params[..., 1].unsqueeze(1)  # size Bx1xM
    a3 = shape_params[..., 2].unsqueeze(1)  # size Bx1xM
    e1 = epsilons[..., 0].unsqueeze(1)  # size Bx1xM
    e2 = epsilons[..., 1].unsqueeze(1)  # size Bx1xM

    # Add a small constant to points that are completely dead center to avoid
    # numerical issues in computing the gradient
    # zeros = X == 0
    # X[zeros] = X[zeros] + 1e-6
    X = ((X > 0).float() * 2 - 1) * torch.max(torch.abs(X), X.new_tensor(1e-6))

    F = ((X[..., 0] / a1) ** 2) ** (1.0 / e2)
    F = F + ((X[..., 1] / a2) ** 2) ** (1.0 / e2)
    F = F ** (e2 / e1)
    F = F + ((X[..., 2] / a3) ** 2) ** (1.0 / e1)

    # Sanity check to make sure that we have the expected size
    assert F.shape == (B, N, M)
    return F**e1


def get_implicit_surface_sq(
    X: torch.Tensor,
    translations: torch.Tensor,
    rotations: torch.Tensor,
    alphas: torch.Tensor,
    epsilons: torch.Tensor,
    sharpness_inside: float = 10.0,
    sharpness_outside: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        X (torch.Tensor): Tensor with size BxNx3, containing the 3D points, for which
            we want to compute the implicit_surface_function. B is the batch size and N
            is the number of points.
        translations (torch.Tensor): Tensor with size BxMx3, containing the translation
            vectors for the M primitives.
        rotations (torch.Tensor): Tensor with size BxMx4 containing the 4 quaternion
            values for the M primitives.
        alphas (torch.Tensor): Tensor with size BxMx3 containing the size along each axis for
            the M primitives.
        epsilons (torch.Tensor): Tensor with size BxMx2 containing the shape parameter along
            each axis for the M primitives.
        sharpness_inside (float, Optional): The sharpness parameter used for the positive points.
            Defaults to 10.
        sharpness_outside (float, Optional): The sharpness parameter used for the negative points.
            Defaults to 10.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the Tensor with size BxNxM of
            the implicit surface values for each primitive M and point N, and the Tensor with
            size BxNxMx3 of the N points transformed in the M primitive centric coordinate systems.
    """
    B, N, _ = X.shape
    _, M, _ = translations.shape

    F, X_transformed = inside_outside_function_from_world_centric_points_sq(
        X, translations, rotations, alphas, epsilons
    )
    mask_inside = (F < F.new_tensor(1.0)).float()
    F_bar = 1.0 - F

    F_inside = torch.sigmoid(sharpness_inside * F_bar) * mask_inside
    F_out = torch.sigmoid(sharpness_outside * F_bar) * (1 - mask_inside)
    F_bar = F_inside + F_out
    assert F_bar.shape == (B, N, M)

    return F_bar, X_transformed


def get_implicit_surface_from_inside_outside_function(
    F: torch.Tensor,
    sharpness_inside: float = 10.0,
    sharpness_outside: float = 10.0,
) -> torch.Tensor:
    """
    Args:
        F (torch.Tensor): Tensor with size BxNxM, containing the inside-outside function
            for each of the 3D points. B is the batch size, N is the number of points and
            M is the number of primitives.
        sharpness_inside (float, Optional): The sharpness parameter used for the positive points.
            Defaults to 10.
        sharpness_outside (float, Optional): The sharpness parameter used for the negative points.
            Defaults to 10.

    Returns:
        torch.Tensor: A Tensor with size BxNxM of the implicit surface values for each primitive M.
    """
    B, N, M = F.shape
    F_bar = 1.0 - F

    F_bar = apply_sigmoid_to_inside_outside_function(
        F_bar, sharpness_inside=sharpness_inside, sharpness_outside=sharpness_outside
    )
    assert F_bar.shape == (B, N, M)

    return F_bar


def apply_sigmoid_to_inside_outside_function(
    F: torch.Tensor,
    sharpness_inside: float = 10.0,
    sharpness_outside: float = 10.0,
) -> torch.Tensor:
    """
    Args:
        F (torch.Tensor): Tensor with size BxNxM, containing the inside-outside function
            for each of the 3D points. The assumption is that the point is inside when F > 0 and
            outside when F <= 0. B is the batch size, N is the number of points and M is the number
            of primitives.
        sharpness_inside (float, Optional): The sharpness parameter used for the positive points.
            Defaults to 10.
        sharpness_outside (float, Optional): The sharpness parameter used for the negative points.
            Defaults to 10.

    Returns:
        torch.Tensor: A Tensor with size BxNxM of the implicit surface values for each primitive M.
    """
    mask_inside = (F > F.new_tensor(0.0)).float()
    F_inside = torch.sigmoid(sharpness_inside * F) * mask_inside
    F_out = torch.sigmoid(sharpness_outside * F) * (1 - mask_inside)
    F_bar = F_inside + F_out
    return F_bar
