import math
from itertools import product

import numpy as np
from pyrr import Matrix44

from part_nerf.dataset.camera import Camera


def get_camera_origins(
    distance: float,
    azimuth_start: int = 0,
    azimuth_stop: int = 375,
    azimuth_step: int = 40,
    elevation_start: int = 60,
    elevation_stop: int = -31,
    elevation_step: int = -20,
    up: str = "y",
):
    def set_camera_location(elevation, azimuth, distance):
        # set location
        x = 1 * math.cos(math.radians(-azimuth))
        x *= math.cos(math.radians(elevation)) * distance
        y = 1 * math.sin(math.radians(-azimuth))
        y *= math.cos(math.radians(elevation)) * distance
        z = 1 * math.sin(math.radians(elevation)) * distance
        if up == "y":
            camera_position = np.array([x, z, y])
        elif up == "z":
            camera_position = np.array([x, y, z])

        return camera_position

    # Define the range of rotation and elevation angles
    azimuth = [xi for xi in range(azimuth_start, azimuth_stop, azimuth_step)]
    elevation = [yi for yi in range(elevation_start, elevation_stop, elevation_step)]
    angles = [xi for xi in product(elevation, azimuth)]

    # Collect all ray_origins from all views
    ray_origins = []
    for elevation, azimuth in angles:
        camera_position = set_camera_location(
            elevation=elevation, azimuth=azimuth, distance=distance
        )
        ray_origins.append(camera_position)
    return np.stack(ray_origins, axis=0)


def get_camera_object(ray_origin: np.ndarray, H: int, W: int, up: str = "y") -> Camera:
    M = Matrix44.perspective_projection(45.0, float(W) / H, 0.1, 1000)
    cx = W / 2
    cy = H / 2
    fx = M[0, 0] * cx
    fy = M[1, 1] * cy
    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1
    # OpenGL uses column vectors hence the transpose and taking the 3rd row as
    # the translation vector
    if up == "y":
        R = Matrix44.look_at(ray_origin, (0, 0, 0.0), (0, 1, 0.0))
    elif up == "z":
        R = Matrix44.look_at(ray_origin, (0, 0, 0.0), (0, 0.0, 1.0))
    Ri = R.inverse
    R = np.asarray(R[:3, :3].T, dtype=np.float32)
    t = np.asarray(Ri[3, :3], dtype=np.float32)
    return Camera(K=K, R=R.T, t=t)


def get_ray_samples(
    ray_origin: np.ndarray,
    H: int,
    W: int,
    near: float,
    far: float,
    num_samples: int,
    up: str = "y",
):
    camera = get_camera_object(ray_origin, H, W, up=up)
    idxs_rows = np.repeat(np.arange(H), W, axis=0)
    idxs_cols = np.tile(np.arange(W), H)
    rays_dict = camera.cast_rays_from_indices(
        H,
        W,
        idxs_rows=idxs_rows,
        idxs_cols=idxs_cols,
        N_samples=num_samples,
        near=near,
        far=far,
    )
    rays_dict["H"], rays_dict["W"] = H, W
    rays_dict["sampled_rows"], rays_dict["sampled_cols"] = idxs_rows, idxs_cols
    return rays_dict
