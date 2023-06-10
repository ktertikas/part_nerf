from typing import Dict, Tuple, Union

import numpy as np
import torch


def to_homogenous(x: torch.Tensor) -> torch.Tensor:
    """Add a 1 at the last dimension."""
    s = x.shape[:-1] + (1,)
    return torch.cat([x, x.new_ones(1).expand(*s)], dim=-1)


def homogenous_dot(P: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = torch.einsum("ij,...j->...i", P, to_homogenous(x))
    return x[..., :-1] / x[..., -1:]


class Camera:
    """Camera implements a pinhole camera defined by matrices K, R, t.

    see "Multiple View Geometry in Computer Vision" by R. Hartley and A.
    Zisserman for notation.

    Args:
        K (torch.Tensor, optional): A 3x3 torch.tensor containing the intrinsic
            camera parameters.
        R (torch.Tensor, optional): A 3x3 torch.tensor containing the rotation
            matrix from camera to world coordinates.
        t (torch.Tensor, optional): A 3x1 torch.tensor containing the translation
            matrix from camera to world coordinates.
        original_W (int, optional): The width of the image based on which the camera's
            principal point was computed.
        original_H (int, optional): The height of the image based on which the camera's
            principal point was computed.
    """

    def __init__(
        self,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
        original_W: int = None,
        original_H: int = None,
    ):
        self._K = K

        # We assume that R is the rotation matrix from camera to world coordinates,
        # namely p_world = p_camera * R
        self._R = R
        self._t = t
        self._original_W = original_W or self._K[0][2] * 2
        self._original_H = original_H or self._K[1][2] * 2

        # A 4x4 projection matrix from the world coordinates to the camera
        # coordinates
        self._world_to_cam = None
        # A 4x4 projection matrix from camera to world coordinates
        self._cam_to_world = None

    @property
    def K(self) -> torch.Tensor:
        # Make sure that K will be a tensor
        if not torch.is_tensor(self._K):
            self._K = torch.tensor(self._K)
        return self._K

    @property
    def R(self) -> torch.Tensor:
        # Make sure that K will be a tensor
        if not torch.is_tensor(self._R):
            self._R = torch.tensor(self._R)
        return self._R

    @property
    def t(self) -> torch.Tensor:
        # Make sure that K will be a tensor
        if not torch.is_tensor(self._t):
            self._t = torch.tensor(self._t)
        return self._t

    @property
    def cam_to_world(self) -> torch.Tensor:
        """Projection matrix that is used to project a 3D point in camera
        coordinates to world coordinates. Note that we assume that the input 3D
        point is in world coordinates.
        """
        if self._cam_to_world is None:
            # self.R is the rotation matrix from camera to world coordinates
            # Let X_w a point in homogenous coordinates in world coordinates
            # and X_c its projection in camera coordinates. We assume the
            # following transformations: X_w = self.R @ X_c + self.t
            self._cam_to_world = torch.cat(
                [
                    torch.cat([self.R, self.t[:, None]], dim=1),
                    torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                ],
                dim=0,
            )
        return self._cam_to_world

    @property
    def world_to_cam(self) -> torch.Tensor:
        """Projection matrix that can be used a 3D point in world coordinates
        to camera coordinates. Note that the point has to be in homogenous
        coordinates.
        """
        if self._world_to_cam is None:
            # self.R is the rotation matrix from camera to world coordinates
            # Let X_w a point in homogenous coordinates in world coordinates
            # and X_c its projection in camera coordinates. We assume the
            # following transformations: X_c = self.R.t() @ (X_w - self.t)
            # Note that we could also simply take the inverse of the
            # self.cam_to_world but I think rewritting this makes things more
            # clear.
            self._world_to_cam = torch.cat(
                [
                    torch.cat([self.R.t(), -self.R.t() @ self.t[:, None]], dim=1),
                    torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                ],
                dim=0,
            )
        return self._world_to_cam

    def get_rays_from_image(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get ray origins and directions from a pinhole camera.

        Code adapted from
        https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py#L123

        Args:
            H (int): Height of image in pixels.
            W (int): Width of image in pixels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The ray origins and directions.
        """
        # Create a grid of pixels
        # WARNING: The meshgrid function in PyTorch will eventually change, so this needs caution
        v, u = torch.meshgrid(
            torch.linspace(0, self._original_W - 1, W),
            torch.linspace(0, self._original_H - 1, H),
        )

        # Convert the pixels [u, v] in raster space to the pixels [xc, yc] camera
        # space using the intrinsic camera matrix K as follows:
        #
        # |u|   | fx 0  ox ||xc|
        # |v| = | 0  fy oy ||yc|
        # |1|   | 0  0  1  ||1|
        #
        # namely,
        # xc = (u - ox) / fx
        # yc = (v - oy) / fy
        #
        # K[0][2] and K[1][2]: camera's principal point (ox, oy)
        # K[0][0] and K[1][1]: camera's focal length (fx, fy)
        fx, fy = self.K[0][0], self.K[1][1]
        ox, oy = self.K[0][2], self.K[1][2]
        # Note that the film's image depicts a mirrored version of reality. To
        # fix this, we'll use a "virtual image" instead of the film itself. The
        # virtual image has the same properties as the film image, but unlike
        # the true image, the virtual image appears in front of the camera, and
        # the projected image is unflipped, i.e the y-axis points upwards.
        # For more details check https://ksimek.github.io/2013/08/13/intrinsic/
        pts_screen = torch.stack(
            [
                (u - ox) / fx,
                -(v - oy) / fy,  # - yc in order for the direction of y to be upwards
                -torch.ones_like(u),
            ],
            dim=-1,
        )

        # Rotate ray directions from camera frame to the world frame
        # rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
        ray_directions = torch.einsum("ijk,lk->ijl", [pts_screen, self.R])
        # Translate camera frame's origin to the world frame. It is the origin of
        # all rays.
        ray_origins = self.t[None, None, :].expand_as(ray_directions)

        return ray_origins, ray_directions

    def compute_z_vals(
        self,
        N_samples: int,
        num_rays: int,
        near: Union[float, torch.Tensor] = 2.0,
        far: Union[float, torch.Tensor] = 6.0,
        lindisp: bool = False,
        perturb: float = 0.0,
        rand: bool = True,
    ) -> torch.Tensor:
        """Sampling depth values along each ray.

        Args:
            N_samples (int): The number of point samples per ray.
            num_rays (int): The number of rays.
            near (Union[float, torch.Tensor], optional): The nearest distance for a ray. Defaults to 2.0.
            far (Union[float, torch.Tensor], optional): The farthest distance for a ray. Defaults to 6.0.
            lindisp (bool, optional): If true, sample linearly in inverse depth. Defaults to False.
            perturb (float, optional): If non-zero, each ray is sampled at stratified random points in time. Defaults to 0.0.
            rand (bool, optional): If True, points are randomly sampled within each sample bin. Defaults to True.

        Returns:
            torch.Tensor: The depth values along each ray.
        """
        if rand:
            # Compute 3D query points
            z_vals = torch.linspace(near, far, steps=N_samples)
            z_vals = z_vals + torch.rand(num_rays, N_samples) * (far - near) / N_samples
        else:
            # Decide where to sample along each ray. Under the logic, all rays will be
            # sampled at the same times.
            t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
            if not lindisp:
                # Space integration times linearly between 'near' and 'far'. Same
                # integration points will be used for all rays.
                z_vals = near * (1.0 - t_vals) + far * t_vals
            else:
                # Sample linearly in inverse depth (disparity).
                z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))
            z_vals = z_vals + torch.zeros(num_rays, N_samples)
            if perturb > 0.0:
                # Get intervals between samples
                mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # Stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape)
                z_vals = lower + (upper - lower) * t_rand
        return z_vals

    def compute_ray_points(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        z_vals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the 3D points along each ray.

        Args:
            ray_origins (torch.Tensor): The origins of each ray.
            ray_directions (torch.Tensor): The directions of each ray.
            z_vals (torch.Tensor): The depth values along each ray.

        Returns:
            torch.Tensor: The 3D points along each ray.
        """
        # Compute 3D query points
        ray_points = (
            ray_origins[..., None, :]
            + ray_directions[..., None, :] * z_vals[..., :, None]
        )
        return ray_points

    def cast_rays(
        self,
        H: int,
        W: int,
        N_samples: int,
        near: Union[float, torch.Tensor] = 2.0,
        far: Union[float, torch.Tensor] = 6.0,
        lindisp: bool = False,
        perturb: float = 0.0,
        rand: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Cast HxW rays from a camera with intrinsic matrix K and camera-to-world
        transformation matrix Rt and and sample N_samples points across each ray

        Code adapted from
        https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L260
        and
        https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L48

        Args:
            H (int): Height of image in pixels.
            W (int): Width of image in pixels.
            N_samples (int): The number of point samples per ray.
            near (Union[float, torch.Tensor], optional): The nearest distance for a ray. Defaults to 2.0.
            far (Union[float, torch.Tensor], optional): The farthest distance for a ray. Defaults to 6.0.
            lindisp (bool, optional): If true, sample linearly in inverse depth. Defaults to False.
            perturb (float, optional): If non-zero, each ray is sampled at stratified random points in time. Defaults to 0.0.
            rand (bool, optional): If True, points are randomly sampled within each sample bin. Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the ray origins, directions, points
                and lengths.
        """
        # Get the ray origins and directions
        ray_origins, ray_directions = self.get_rays_from_image(H, W)
        num_rays = H * W
        z_vals = self.compute_z_vals(
            N_samples,
            num_rays,
            near=near,
            far=far,
            lindisp=lindisp,
            perturb=perturb,
            rand=rand,
        )

        ray_points = self.compute_ray_points(ray_origins, ray_directions, z_vals)

        return {
            "ray_origins": ray_origins,
            "ray_directions": ray_directions,
            "ray_points": ray_points,
            "ray_lengths": z_vals,
        }

    def cast_rays_from_indices(
        self,
        H: int,
        W: int,
        N_samples: int,
        idxs_rows: np.ndarray,
        idxs_cols: np.ndarray,
        near: Union[float, torch.Tensor] = 2.0,
        far: Union[float, torch.Tensor] = 6.0,
        lindisp: bool = False,
        perturb: float = 0.0,
        rand: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Cast HxW rays from a camera with intrinsic matrix K and camera-to-world
        transformation matrix Rt and and sample N_samples points across each ray.
        We provide the rows indices directly to speed up computation when the image
        size is significant.

        Code adapted from
        https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L260
        and
        https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L48

        Args:
            H (int): Height of image in pixels.
            W (int): Width of image in pixels.
            N_samples (int): The number of point samples per ray.
            inds_rows (np.ndarray): The indices for selection of pixels in the row space.
            inds_cols (np.ndarray): The indices for selection of pixels in the column space.
            near (Union[float, torch.Tensor], optional): The nearest distance for a ray. Defaults to 2.0.
            far (Union[float, torch.Tensor], optional): The farthest distance for a ray. Defaults to 6.0.
            lindisp (bool, optional): If true, sample linearly in inverse depth. Defaults to False.
            perturb (float, optional): If non-zero, each ray is sampled at stratified random points in time. Defaults to 0.0.
            rand (bool, optional): If True, points are randomly sampled within each sample bin. Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the ray origins, directions, points
                and lengths.
        """
        # Get the ray origins and directions
        ray_origins, ray_directions = self.get_rays_from_image(H, W)
        assert len(idxs_rows) == len(idxs_cols)
        num_rays = len(idxs_rows)
        # selecting indices before large computations
        ray_origins = ray_origins[idxs_rows, idxs_cols, ...]
        ray_directions = ray_directions[idxs_rows, idxs_cols, ...]
        z_vals = self.compute_z_vals(
            N_samples,
            num_rays,
            near=near,
            far=far,
            lindisp=lindisp,
            perturb=perturb,
            rand=rand,
        )

        ray_points = self.compute_ray_points(ray_origins, ray_directions, z_vals)

        return {
            "ray_origins": ray_origins,
            "ray_directions": ray_directions,
            "ray_points": ray_points,
            "ray_lengths": z_vals,
        }

    def project_points(
        self, points: torch.Tensor, projection_map: str = "world_to_pix"
    ) -> torch.Tensor:
        """Given a np.array of 3D points in world coordinates, project them to
        using an affine transformation on homogenous coordinates.
        """
        # Make sure that the projection matrix has the correct size
        if projection_map == "world_to_pix":
            points_cam = to_homogenous(points) @ self.world_to_cam.t()
            fx, fy = self.K[0][0], self.K[1][1]
            ox, oy = self.K[0][2], self.K[1][2]
            u = points_cam[..., 0] * fx / (-points_cam[..., 2]) + ox
            v = -points_cam[..., 1] * fy / (-points_cam[..., 2]) + oy
            return torch.stack([u, v], dim=-1)

        if projection_map == "world_to_cam":
            P = self.world_to_cam
        elif projection_map == "cam_to_world":
            P = self.cam_to_world
        else:
            raise NotImplementedError()

        return homogenous_dot(P, points)
