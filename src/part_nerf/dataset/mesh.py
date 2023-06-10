from pathlib import Path

import numpy as np
import trimesh


class Mesh:
    def __init__(self, mesh: trimesh.Trimesh, normalize: bool = False):
        self.mesh = mesh
        # Normalize points such that they are in the unit cube
        if normalize:
            bbox = self.mesh.bounding_box.bounds
            # Compute location and scale
            loc = (bbox[0] + bbox[1]) / 2
            scale = (bbox[1] - bbox[0]).max()  # / (1 - 0.05)

            # Transform input mesh
            self.mesh.apply_translation(-loc)
            self.mesh.apply_scale(1 / scale)

        # Make sure that the input meshes are watertight
        assert self.mesh.is_watertight, f"Mesh is not watertight!"

        self._vertices = None
        self._vertex_normals = None
        self._faces = None
        self._face_normals = None

    @property
    def vertices(self) -> np.ndarray:
        if self._vertices is None:
            self._vertices = np.array(self.mesh.vertices)
        return self._vertices

    @property
    def vertex_normals(self) -> np.ndarray:
        if self._vertex_normals is None:
            self._vertex_normals = np.array(self.mesh.vertex_normals)
        return self._vertex_normals

    @property
    def faces(self) -> np.ndarray:
        if self._faces is None:
            self._faces = np.array(self.mesh.faces)
        return self._faces

    @property
    def face_normals(self) -> np.ndarray:
        if self._face_normals is None:
            self._face_normals = np.array(self.mesh.face_normals)
        return self._face_normals

    def sample_faces(self, N=10000):
        P, t = trimesh.sample.sample_surface(self.mesh, N)
        return np.hstack([P, self.face_normals[t, :]])

    @classmethod
    def from_file(cls, filename, normalize):
        return cls(trimesh.load(filename, process=False), normalize)


def read_mesh_file(filename: Path, normalize: bool) -> Mesh:
    return Mesh.from_file(filename, normalize)
