"""Script for rendering ShapeNet data and generating a dataset that can
be used for taining NeRF models. The rendering is done using multiprocessing
for speedup.
"""
import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from pyvirtualdisplay import Display
from simple_3dviz import Mesh
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.scenes import Scene
from simple_3dviz.utils import save_frame
from tqdm import tqdm


class ShapeNetModel:
    def __init__(self, base_dir, model_tag):
        self._base_dir = base_dir
        self._model_tag = model_tag

    @property
    def raw_model_path(self):
        return os.path.join(self._base_dir, self._model_tag, "model.obj")

    @property
    def textured_mesh(self):
        try:
            return TexturedMesh.from_file(self.raw_model_path)
        except:
            print(f"Loading model {self.raw_model_path} failed", flush=True)
            return None

    @property
    def mesh(self):
        try:
            return Mesh.from_file(self.raw_model_path)
        except:
            # print(f"Loading model {self.raw_model_path} failed", flush=True)
            return None


class ShapeNetObjectsDataset:
    def __init__(self, base_dir):
        self._base_dir = base_dir
        self._model_paths = [
            os.path.join(self._base_dir, oi)
            for oi in os.listdir(self._base_dir)
            if os.path.exists(os.path.join(self._base_dir, oi))
        ]

    def __len__(self):
        return len(self._model_paths)

    def __getitem__(self, idx):
        return ShapeNetModel(self._base_dir, self._model_paths[idx])


def scene_from_args(args):
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size, background=args.background)
    scene.up_vector = args.up_vector
    if "camera_target" in args:
        scene.camera_target = args.camera_target
    if "camera_position" in args:
        scene.camera_position = args.camera_position
        scene.light = args.camera_position
    return scene


def render_scene(
    scene,
    camera_position,
    camera_target,
    up_vector,
    background,
    path_to_file,
    H=256,
    W=256,
):
    # Now we want to construct the camera matrix. The only thing we need
    # Place the camera
    scene.camera_position = camera_position
    scene.camera_target = camera_target
    scene.up_vector = up_vector
    scene.light = camera_position
    scene.background = background

    # Following this nice example from
    # https://programtalk.com/python-more-examples/pyrr.Matrix44/
    #
    # it seems that the projection matrix contructed from pyrr's
    # Matrix44.perspective_projection function has the following format
    #
    # projection = Matrix44([
    #   [fx/cx, 0, 0, 0],
    #   [0, fy/cy, 0, 0],
    #   [0, 0, (-zfar - znear)/(zfar - znear), -1],
    #   [0, 0, (-2.0*zfar*znear)/(zfar - znear), 0]
    # ])
    #
    # Note that this matrix is equivalent to the matrix produced by glFrustrum
    # For more details please see here
    # http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    #
    # Assuming that cx = W / 2 and cy = H /2
    # we now get the camera intrinsic matrix
    projection = scene._camera
    cx = W / 2
    cy = H / 2
    fx = projection[0, 0] * cx
    fy = projection[1, 1] * cy
    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1

    world_to_cam = scene.mv
    R_world_to_cam = np.asarray(world_to_cam[:3, :3].T)
    translation = np.asarray(scene.vm[3, :3])

    # Render the scene
    scene.render()
    save_frame(path_to_file, scene.frame)

    return K, R_world_to_cam, translation


def set_camera_location(elevation, azimuth, distance):
    # set location
    x = (
        1
        * math.cos(math.radians(-azimuth))
        * math.cos(math.radians(elevation))
        * distance
    )
    y = (
        1
        * math.sin(math.radians(-azimuth))
        * math.cos(math.radians(elevation))
        * distance
    )
    z = 1 * math.sin(math.radians(elevation)) * distance

    camera_position = np.array([x, z, y])
    return camera_position


def render_images(datum, angles, args, scene):
    # Define the paths to save the images, the camera and the correspondatumng
    # masks after rendering each object
    path_to_images_dir = os.path.join(
        datum._model_tag, f"images_{args.window_size[0]}_{args.window_size[1]}"
    )
    path_to_cameras_dir = os.path.join(
        datum._model_tag, f"cameras_{args.window_size[0]}_{args.window_size[1]}"
    )
    path_to_masks_dir = os.path.join(
        datum._model_tag, f"masks_{args.window_size[0]}_{args.window_size[1]}"
    )
    # Check optimistically if the paths already exist and if everything was
    # previously rendered
    if (
        os.path.exists(path_to_images_dir)
        and len(os.listdir(path_to_images_dir)) == len(angles)
        and os.path.exists(path_to_cameras_dir)
        and len(os.listdir(path_to_cameras_dir)) == len(angles)
        and os.path.exists(path_to_masks_dir)
        and len(os.listdir(path_to_masks_dir)) == len(angles)
    ):
        return

    # Check if these paths exist and it they don't create them
    if not os.path.exists(path_to_images_dir):
        os.makedirs(path_to_images_dir)
    if not os.path.exists(path_to_cameras_dir):
        os.makedirs(path_to_cameras_dir)
    if not os.path.exists(path_to_masks_dir):
        os.makedirs(path_to_masks_dir)

    renderable = datum.textured_mesh
    if renderable is None:
        print(datum._model_tag)
        return
    renderable.to_unit_cube()

    try:
        scene.add(renderable)
    except Exception as e:
        print(f"Skipping model: {datum._model_tag}")
        print(f"Type of exception raised: {type(e).__name__}")

    # Start rendering the images
    for i, (elevation, azimuth) in enumerate(angles):
        camera_position = set_camera_location(
            elevation=elevation, azimuth=azimuth, distance=args.camera_distance
        )
        K, R_world_to_cam, translation = render_scene(
            scene,
            camera_position=camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            path_to_file=os.path.join(path_to_images_dir, f"{i:05}.png"),
            W=args.window_size[0],
            H=args.window_size[1],
        )
        np.savez(
            os.path.join(path_to_cameras_dir, f"{i:05}"),
            R_world_to_cam=R_world_to_cam,
            R_cam_to_world=R_world_to_cam.T,
            K=K,
            translation=translation,
        )

    # Clear the scene in the end
    scene.clear()

    # Start rendering the object masks
    mesh = datum.mesh
    mesh.to_unit_cube()
    mesh.mode = "flat"
    mesh.colors = (1.0, 1.0, 1.0)
    scene.add(mesh)
    for i, (elevation, azimuth) in enumerate(angles):
        camera_position = set_camera_location(
            elevation=elevation, azimuth=azimuth, distance=args.camera_distance
        )
        _, _, _ = render_scene(
            scene,
            camera_position=camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=(0.0, 0.0, 0.0, 1.0),
            path_to_file=os.path.join(path_to_masks_dir, f"{i:05}.png"),
            W=args.window_size[0],
            H=args.window_size[1],
        )
    # Clear the scene in the end
    scene.clear()


def main(argv):
    parser = argparse.ArgumentParser(description="Render the ShapeNet dataset")
    parser.add_argument(
        "dataset_directory", help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene",
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,0",
        help="Up vector of the scene",
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="256,256",
        help="Define the size of the scene and the window",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Define the number of cpus for multiprocessing",
    )
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=1.5,
        help="Define the distance of the camera to the origin point",
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Point that the camera is looking at",
    )
    args = parser.parse_args(argv)

    dataset = ShapeNetObjectsDataset(args.dataset_directory)
    print(f"ShapeNet dataset contains {len(dataset)} objects...")

    # Define the range of rotation and elevation angles
    azimuth = [xi for xi in range(0, 375, 15)]
    elevation = [yi for yi in range(-30, 75, 10)]
    # Gather all the angles used for rendering each object
    angles = [xi for xi in product(elevation, azimuth)]

    num_processes = args.num_cpus
    with Display() as disp:
        scenes_list = [scene_from_args(args) for i in range(num_processes)]
        with ProcessPoolExecutor(num_processes) as executor:
            with tqdm(total=len(dataset)) as progress_bar:
                for i, datum in enumerate(dataset):
                    future = executor.submit(
                        render_images(
                            datum, angles, args, scenes_list[i % num_processes]
                        )
                    )
                    future.add_done_callback(lambda p: progress_bar.update())


if __name__ == "__main__":
    main(sys.argv[1:])
