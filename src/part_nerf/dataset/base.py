from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .camera import Camera
from .index_samplers import (
    get_all_indices,
    get_all_positive_indices,
    get_equal_indices,
    get_uniform_indices,
)
from .mesh import Mesh, read_mesh_file


class BaseFrameDatum:
    """Implements a simple interface for all scenes, independent of dataset.
    Every frame has a unique tag and is represented by one image accompanied
    by its corresponding camera pose and optionally a 2D Mask and a Mesh object
    that contains its 3D representation.
    """

    def __init__(self, tag: str, config: Dict):
        self._tag = tag
        self._config = config

        # Initialize the contents of this instance to empty so that they can be
        # lazy loaded
        self._mesh = None
        self._image_path = None
        self._camera_path = None
        self._image_dims = None
        self._original_image_dims = None

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def config(self) -> Dict:
        return self._config

    @property
    def path_to_mesh_file(self) -> Path:
        raise NotImplementedError()

    @property
    def path_to_img_file(self) -> Path:
        raise NotImplementedError()

    @property
    def path_to_camera_file(self) -> Path:
        raise NotImplementedError()

    @property
    def path_to_mask_file(self) -> Path:
        raise NotImplementedError()

    @property
    def images_dir(self) -> Path:
        return self.path_to_img_file.parent

    @property
    def cameras_dir(self) -> Path:
        return self.path_to_camera_file.parent

    @property
    def masks_dir(self) -> Path:
        return self.path_to_mask_file.parent

    @property
    def groundtruth_mesh(self) -> Mesh:
        if self._mesh is None:
            self._mesh = read_mesh_file(
                self.path_to_mesh_file, self.config.get("normalize", True)
            )
        return self._mesh

    def get_mask(self) -> Optional[torch.Tensor]:
        mask_size = tuple(self.config.get("image_size"))
        if self.path_to_mask_file.is_file():
            mask = Image.open(self.path_to_mask_file).convert("L")
            mask = mask.resize(
                mask_size,
                resample=Image.BILINEAR,
            )
            mask_np = np.asarray(mask, dtype=bool)
            return torch.from_numpy(mask_np.astype(np.float32))
        # if mask file does not exist, check existence of alpha channel!
        img = Image.open(self.path_to_img_file)
        if img.mode == "RGBA":
            mask = img.split()[-1]
            mask = mask.resize(
                mask_size,
                resample=Image.BILINEAR,
            )
            mask_np = np.asarray(mask, dtype=bool)
            return torch.from_numpy(mask_np.astype(np.float32))

    def get_image(self) -> torch.Tensor:
        # If the option of white background is False, we will try to
        # remove the background by using the mask. If the mask does not
        # exist, we will simply return the image as is.
        white_background = self.config.get("white_background", False)
        img = Image.open(self.path_to_img_file).convert("RGB")
        img_size = tuple(self.config.get("image_size"))
        img = img.resize(
            img_size,
            resample=Image.BILINEAR,
        )
        torch_img = torch.from_numpy(
            np.asarray(img).astype(np.float32) / np.float32(255)
        )
        torch_mask = self.get_mask()
        if torch_mask is not None:
            # using mask turn all outer values to 0
            torch_img = torch_img * torch_mask[..., None]
            if white_background:
                # color outer mask rays to white
                torch_img[~torch_mask] = 1.0
        return torch_img

    @property
    def image_dims(self) -> Tuple[int, int]:
        if self._image_dims is None:
            # Load the image and compute its size
            img = self.get_image()
            self._image_dims = (img.shape[0], img.shape[1])
        return self._image_dims

    @property
    def original_image_dims(self) -> Tuple[int, int]:
        if self._original_image_dims is None:
            img = Image.open(self.path_to_img_file)
            W, H = img.size
            self._original_image_dims = (H, W)
        return self._original_image_dims

    def get_camera(self) -> Camera:
        camera = np.load(self.path_to_camera_file, allow_pickle=True)
        original_H, original_W = self.original_image_dims
        r_name = "R" if "R" in list(camera.keys()) else "R_cam_to_world"
        t_name = "t" if "t" in list(camera.keys()) else "translation"
        return Camera(
            K=camera["K"].astype(np.float32),
            R=camera[r_name].astype(np.float32),
            t=camera[t_name].astype(np.float32),
            original_H=original_H,
            original_W=original_W,
        )

    def get_rays_from_indices(
        self, idxs_rows: np.ndarray, idxs_cols: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        camera = self.get_camera()
        H, W = self.image_dims
        rays = camera.cast_rays_from_indices(
            H,
            W,
            idxs_rows=idxs_rows,
            idxs_cols=idxs_cols,
            N_samples=self.config.get("n_samples", 32),
            near=self.config.get("near", 2.0),
            far=self.config.get("far", 6.0),
            lindisp=self.config.get("lindisp", False),
            perturb=self.config.get("perturb", 0.0),
            rand=self.config.get("rand", True),
        )
        return {**rays}

    def _sample_indices_equal(
        self, N: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Only to be used when ground truth mask is available!
        H, W = self.image_dims  # mask and image have the same dimensions
        mask = np.asarray(
            self.get_mask(), dtype=np.float32
        )  # needed as by default mask is torch Tensor
        flat_mask = mask.ravel()  # (H * W,)
        return get_equal_indices(N, H, W, flat_mask)

    def _sample_indices_positive(
        self, N: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Only get positive rays!
        # Only to be used when ground truth mask is available!
        H, W = self.image_dims  # mask and image have the same dimensions
        mask = np.asarray(
            self.get_mask(), dtype=np.float32
        )  # needed as by default mask is torch Tensor

        flat_mask = mask.ravel()  # (H * W,)
        # Take all rows and columns
        idxs_rows, idxs_cols, weights = get_all_positive_indices(H, W, flat_mask)

        if N > 0:
            # Oversampling of positive indices. If N=-1, no oversampling
            # WARNING: when N=-1, batching will have issues as the number of indices
            # will vary.
            selected_idxs = np.random.choice(len(idxs_rows), N)
            idxs_rows = idxs_rows[selected_idxs]
            idxs_cols = idxs_cols[selected_idxs]
            weights = weights[selected_idxs]

        return idxs_rows, idxs_cols, weights

    def _sample_indices_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = self.image_dims
        # Simply take all rows and columns
        return get_all_indices(H, W)

    def _sample_indices_uniform(
        self, N: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert N > 0
        H, W = self.image_dims
        return get_uniform_indices(N, H, W)

    def get_sample_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = self.config.get("n_rays", -1)
        sampling_type = self.config.get("sampling_type", "uniform")
        if N == -1:
            if sampling_type == "positive":
                idxs_rows, idxs_cols, weights = self._sample_indices_positive(N)
            else:
                idxs_rows, idxs_cols, weights = self._sample_indices_all()
        else:
            sampling_type = self.config.get("sampling_type", "uniform")
            sampler_dict = {
                "uniform": self._sample_indices_uniform,
                "equal": self._sample_indices_equal,
                "positive": self._sample_indices_positive,
            }
            sampler = sampler_dict[sampling_type]
            idxs_rows, idxs_cols, weights = sampler(N)
        return idxs_rows, idxs_cols, weights

    def sample_rays(self) -> Dict[str, Union[int, torch.Tensor]]:
        idxs_rows, idxs_cols, _ = self.get_sample_indices()
        rays_dict = self.get_rays_from_indices(idxs_rows, idxs_cols)
        rays_dict["H"], rays_dict["W"] = self.image_dims
        rays_dict["sampled_rows"] = idxs_rows
        rays_dict["sampled_cols"] = idxs_cols
        return rays_dict

    def sample_ray_color_pairs(self) -> Dict[str, Union[int, torch.Tensor]]:
        idxs_rows, idxs_cols, weights = self.get_sample_indices()
        rays_dict = self.get_rays_from_indices(idxs_rows, idxs_cols)
        colors = self.get_image()

        rays_dict["colors"] = colors[idxs_rows, idxs_cols, ...]
        rays_dict["H"], rays_dict["W"] = self.image_dims
        rays_dict["sampled_rows"] = idxs_rows
        rays_dict["sampled_cols"] = idxs_cols
        rays_dict["weights"] = weights
        return rays_dict

    def sample_ray_color_mask_pairs(self) -> Dict[str, Union[int, torch.Tensor]]:
        rays_dict = self.sample_ray_color_pairs()
        H, W = self.image_dims
        masks = self.get_mask().reshape((H, W, 1))
        idxs_rows, idxs_cols = rays_dict["sampled_rows"], rays_dict["sampled_cols"]
        rays_dict["gt_mask"] = masks[idxs_rows, idxs_cols, ...]
        return rays_dict


class BaseDataCollection:
    """Base Data Collection definition, that serves as a container of multiple BaseDatum
    instances.
    """

    def __len__(self) -> int:
        raise NotImplementedError()

    def _get_datum(self, i) -> BaseFrameDatum:
        raise NotImplementedError()

    def __getitem__(self, i) -> BaseFrameDatum:
        if i >= len(self):
            raise IndexError()
        return self._get_datum(i)


class BaseDataset(Dataset):
    """Pytorch Dataset wrapper for all the datasets"""

    def __init__(
        self, data_collection: BaseDataCollection, transform: Optional[Callable] = None
    ):
        """
        Args:
            data_collection (BaseDataCollection): An object of type BaseDataCollection.
            transform (Callable, optional): A callable that applies a transform to each
                sample. Defaults to None.
        """
        super().__init__()
        self._data_collection = data_collection
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data_collection)

    def __getitem__(self, idx: int) -> Dict:
        datum = self._get_item_inner(idx)
        if self._transform:
            datum = self._transform(datum)
        return datum

    def _get_item_inner(self, idx: int) -> Dict:
        raise NotImplementedError()

    def get_random_datapoint(self):
        return self.__getitem__(np.random.choice(len(self)))

    @property
    def shapes(self):
        raise NotImplementedError()

    @property
    def internal_data_collection(self) -> BaseDataCollection:
        return self._data_collection
