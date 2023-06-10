from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BaseDataCollection, BaseDataset
from .data_collections import DataCollectionBuilder
from .splits_builder import splits_builder_factory


class DatasetUnion(Dataset):
    """PyTorch Dataset Union definition, using a list of BaseDataset objects."""

    def __init__(self, *datasets: List[BaseDataset]):
        self._check_datasets(datasets)
        self._datasets = datasets

    def __len__(self) -> int:
        return len(self._datasets[0])

    def __getitem__(self, idx: int) -> Dict:
        return self._get_item_inner(idx)

    def _get_item_inner(self, idx: int) -> Dict:
        dataset_dict = {}
        for d in self._datasets:
            dataset_dict.update(d[idx])
        return dataset_dict

    def get_random_datapoint(self):
        return self.__getitem__(np.random.choice(len(self)))

    @property
    def shapes(self) -> Dict[str, Tuple[int]]:
        shapes_dict = {d.shapes for d in self._datasets}
        return shapes_dict

    @staticmethod
    def _check_datasets(datasets: List[BaseDataset]):
        assert (
            len(datasets) > 0
        ), "Number of datasets passed inside the DatasetUnion should not be 0."
        data_it = iter(datasets)
        data_len = len(next(data_it))
        assert all(
            len(d) == data_len for d in data_it
        ), "All dataset instances should have the same length."


class ImageOnlyDataset(BaseDataset):
    """Get only images as the output of this dataset."""

    # TODO: Remove the torch transforms, and get PIL Image as output

    def _get_item_inner(self, idx: int) -> Dict[str, torch.Tensor]:
        img = self.internal_data_collection[idx].get_image()
        return {"img": img}

    @property
    def shapes(self) -> Dict[str, Tuple[int]]:
        H, W = tuple(
            map(
                int,
                self.internal_data_collection.data_config.get("image_size").split(","),
            )
        )
        return {"img": (H, W, 3)}


class RaysOnlyDataset(BaseDataset):
    """Get rays dataset, i.e. samples of rays. To be used for NeRF-like experiments,
    mostly targeting generation / inference of new views."""

    def _get_item_inner(self, idx: int) -> Dict[str, Union[int, torch.Tensor]]:
        rays_dict = self.internal_data_collection[idx].sample_rays()
        return rays_dict

    @property
    def shapes(self) -> Dict[str, Tuple[int]]:
        H, W = tuple(
            map(
                int,
                self.internal_data_collection.data_config.get("image_size").split(","),
            )
        )
        N = self.internal_data_collection.data_config.get("n_rays", -1)
        if N != -1:
            n_rays = N
        else:
            n_rays = H * W
        n_samples = self.internal_data_collection.data_config.get("N_samples", 32)
        return {
            "ray_origins": (n_rays, 3),
            "ray_directions": (n_rays, 3),
            "ray_points": (n_rays, n_samples, 3),
            "ray_lengths": (n_rays, n_samples),
            "sampled_rows": (n_rays,),
            "sampled_cols": (n_rays,),
        }


class RaysDataset(BaseDataset):
    """Get rays dataset, i.e. samples of rays and respective colors. To
    be used for NeRF-like experiments."""

    # TODO: Turn to numpy arrays, and let the transformation to torch Tensors
    # happen on the dataset_factory world.

    def _get_item_inner(self, idx: int) -> Dict[str, Union[int, torch.Tensor]]:
        rays_dict = self.internal_data_collection[idx].sample_ray_color_pairs()
        return {**rays_dict}

    @property
    def shapes(self) -> Dict[str, Tuple[int]]:
        H, W = tuple(
            map(
                int,
                self.internal_data_collection.data_config.get("image_size").split(","),
            )
        )
        N = self.internal_data_collection.data_config.get("n_rays", -1)
        if N != -1:
            n_rays = N
        else:
            n_rays = H * W
        n_samples = self.internal_data_collection.data_config.get("N_samples", 32)
        return {
            "ray_origins": (n_rays, 3),
            "ray_directions": (n_rays, 3),
            "ray_points": (n_rays, n_samples, 3),
            "ray_lengths": (n_rays, n_samples),
            "sampled_rows": (n_rays,),
            "sampled_cols": (n_rays,),
            "colors": (n_rays, 3),
        }


class RaysMasksDataset(BaseDataset):
    def _get_item_inner(self, idx: int) -> Dict[str, torch.Tensor]:
        rays_masks_dict = self.internal_data_collection[
            idx
        ].sample_ray_color_mask_pairs()
        return {**rays_masks_dict}

    @property
    def shapes(self) -> Dict[str, Tuple[int]]:
        H, W = tuple(
            map(
                int,
                self.internal_data_collection.data_config.get("image_size").split(","),
            )
        )
        N = self.internal_data_collection.data_config.get("n_rays", -1)
        if N != -1:
            n_rays = N
        else:
            n_rays = H * W
        n_samples = self.internal_data_collection.data_config.get("N_samples", 32)
        return {
            "ray_origins": (n_rays, 3),
            "ray_directions": (n_rays, 3),
            "ray_points": (n_rays, n_samples, 3),
            "ray_lengths": (n_rays, n_samples),
            "sampled_rows": (n_rays,),
            "sampled_cols": (n_rays,),
            "colors": (n_rays, 3),
            "gt_masks": (n_rays,),
        }


class SceneIdDataset(BaseDataset):
    """Get scene ids for each dataset. This is used in the autodecoder case. We explicitly check the number
    of unique scene ids in order to change index ids in case some ids have been filtered out during subset
    selection.

    WARNING: For validation purposes the final validation set needs to contain at least one of the exact
    same paths as the training set in the filtering csv, otherwise the ids will not be parsed properly.
    """

    @property
    def unique_scenes_dict(self) -> Dict[str, int]:
        # pass dataset once to get all available scenes
        if not hasattr(self, "_unique_scenes_dict"):
            self._unique_scenes_dict = self.get_unique_scenes_dict()
        return self._unique_scenes_dict

    def _get_item_inner(self, idx: int) -> Dict:
        scene = self.internal_data_collection[idx].get_scene_tag()
        return {"scene_id": self.unique_scenes_dict[scene]}

    def get_unique_scenes_dict(self):
        # we keep the assumption that categories are sorted by file name
        unique_scenes = set()
        for i in range(len(self.internal_data_collection)):
            scene_tag = self.internal_data_collection[i].get_scene_tag()
            unique_scenes.add(scene_tag)
        unique_scenes_list = sorted(unique_scenes)
        return {scene: idx for idx, scene in enumerate(unique_scenes_list)}

    @property
    def shapes(self) -> Dict[str, int]:
        return {"scene_id": (1,)}


def dataset_factory(name: str, data_collection: BaseDataCollection) -> DatasetUnion:
    image_only = ImageOnlyDataset(data_collection=data_collection, transform=None)
    rays_only = RaysOnlyDataset(data_collection=data_collection, transform=None)
    rays_and_colors = RaysDataset(data_collection=data_collection, transform=None)
    rays_masks_and_colors = RaysMasksDataset(
        data_collection=data_collection, transform=None
    )
    scene_id = SceneIdDataset(data_collection=data_collection, transform=None)
    return {
        "ImageOnlyDataset": DatasetUnion(image_only),
        "RaysDataset": DatasetUnion(rays_only),
        "RaysColorsDataset": DatasetUnion(rays_and_colors),
        "RaysColorsIndexedDataset": DatasetUnion(rays_and_colors, scene_id),
        "RaysMasksColorsIndexedDataset": DatasetUnion(rays_masks_and_colors, scene_id),
        "ImageRaysColorsDataset": DatasetUnion(image_only, rays_and_colors),
    }[name]


def build_dataset(
    config: Dict,
    data_tags: List[str],
    category_tags: List[str],
    keep_splits: List[str],
    random_subset_pct: float = 1.0,
) -> DatasetUnion:
    dataset_directory = config["dataset_directory"]
    collection_type = config["collection_type"]
    train_test_splits_file = config["splits_file"]
    dataset = dataset_factory(
        config["dataset_factory"],
        DataCollectionBuilder(config)
        .with_data_collection(collection_type)
        .filter_train_test(
            split_builder=splits_builder_factory(collection_type)(
                train_test_splits_file
            ),
            keep_splits=keep_splits,
        )
        .filter_tags(data_tags)
        .filter_category_tags(category_tags)
        .random_subset(random_subset_pct)
        .build(dataset_directory),
    )
    return dataset
