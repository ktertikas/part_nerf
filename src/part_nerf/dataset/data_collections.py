from pathlib import Path
from typing import Dict, List

import numpy as np

from .base import BaseDataCollection, BaseFrameDatum
from .splits_builder import SplitsBuilder


class DataCollectionSubset(BaseDataCollection):
    def __init__(self, collection: BaseDataCollection, subset: List[str]):
        self._collection = collection
        self._subset = subset

    def __len__(self) -> int:
        return len(self._subset)

    def _get_sample(self, idx: int) -> BaseFrameDatum:
        return self._collection[self._subset[idx]]

    def __getitem__(self, idx: int) -> BaseFrameDatum:
        if idx >= len(self):
            raise IndexError()
        return self._get_sample(idx)


class CategorySubset(DataCollectionSubset):
    def __init__(self, collection: BaseDataCollection, category_tags: List[str]):
        category_tags = set(category_tags)
        subset = [i for (i, m) in enumerate(collection) if m.category in category_tags]
        super().__init__(collection=collection, subset=subset)


class TagSubset(DataCollectionSubset):
    def __init__(self, collection: BaseDataCollection, tags: List[str]):
        tags = set(tags)
        subset = [i for (i, m) in enumerate(collection) if m.tag in tags]
        super().__init__(collection=collection, subset=subset)


class RandomSubset(DataCollectionSubset):
    def __init__(self, collection: BaseDataCollection, percentage: float):
        N = len(collection)
        subset = np.random.choice(N, int(N * percentage)).tolist()
        super().__init__(collection=collection, subset=subset)


class BasicScenes(BaseDataCollection):
    """This is a generic data collection, that contains a folder per scene.
    Each scene is then structured as follows, one folder for the images, one
    for the camera poses and optionally also the ground-truth mesh of the object.
    """

    class FrameDatum(BaseFrameDatum):
        def __init__(self, base_dir: Path, tag: str, config: Dict):
            super().__init__(tag, config)
            self._base_dir = base_dir
            # Extract the name of the scene and the name of the frame
            self._scene, self._frame = tag.split(":")
            self._mesh_ext = self.config.get("mesh_ext", ".obj")
            self._images_folder = self.config.get("images_folder", "images")
            self._cameras_folder = self.config.get("cameras_folder", "cameras")
            self._masks_folder = self.config.get("masks_folder", "masks")

        @property
        def path_to_mesh_file(self) -> Path:
            mesh_dir = self._base_dir / f"{self._scene}"
            if (mesh_dir / "model.obj").exists():
                return mesh_dir / "model.obj"
            elif (mesh_dir / "model.off").exists():
                return mesh_dir / "model.off"
            else:
                raise FileNotFoundError(
                    f"The provided mesh_path: {mesh_dir} doesn't contain any mesh!"
                )

        @property
        def path_to_img_file(self) -> Path:
            return (
                self._base_dir
                / f"{self._scene}/{self._images_folder}/{self._frame}.png"
            )

        @property
        def path_to_camera_file(self) -> Path:
            return (
                self._base_dir
                / f"{self._scene}/{self._cameras_folder}/{self._frame}.npz"
            )

        @property
        def path_to_mask_file(self) -> Path:
            return (
                self._base_dir / f"{self._scene}/{self._masks_folder}/{self._frame}.png"
            )

        def get_scene_tag(self) -> str:
            return self._scene

    def __init__(self, base_dir: str, config: Dict):
        self._base_dir = Path(base_dir)
        self._config = config
        self._scenes_folders = sorted(
            [d.name for d in self._base_dir.iterdir() if (self._base_dir / d).is_dir()]
        )
        # We associate every image within the images_folder directory with a
        # tag
        self._images_folder = config.get("images_folder", "images")

        tag_sid_list = []
        for d in self._scenes_folders:
            for l in sorted((self._base_dir / d / self._images_folder).iterdir()):
                if l.suffix == ".png":
                    tag_sid_list.append(f"{d}:{l.stem}")
        self._tags = tag_sid_list
        print(f"Found {len(self)} BasicScenes datums")

    @property
    def config(self) -> Dict:
        return self._config

    def __len__(self) -> int:
        return len(self._tags)

    def _get_datum(self, i) -> FrameDatum:
        return self.FrameDatum(self._base_dir, self._tags[i], self.config)


def data_collection_factory(collection_type: str) -> BaseDataCollection:
    return {
        "shapenet": BasicScenes,
        "lego_dataset": BasicScenes,
    }[collection_type]


class DataCollectionBuilder:
    def __init__(self, config: Dict):
        self._config = config

        self._data_collection = None
        self._tags: List[str] = []
        self._category_tags: List[str] = []
        self._train_test_splits: List[str] = []
        self._percentage: float = 1.0

    def with_data_collection(self, collection_type: str):
        self._data_collection = data_collection_factory(collection_type)
        return self

    def filter_tags(self, tags: List[str]):
        self._tags = tags
        return self

    def filter_category_tags(self, tags: List[str]):
        self._category_tags = tags
        return self

    def filter_train_test(self, split_builder: SplitsBuilder, keep_splits: List[str]):
        self._train_test_splits = split_builder.get_splits(keep_splits=keep_splits)
        return self

    def random_subset(self, percentage: float):
        self._percentage = percentage
        return self

    def build(self, base_dir: str) -> BaseDataCollection:
        data_collection = self._data_collection(base_dir, self._config)
        if len(self._train_test_splits) > 0:
            prev_len = len(data_collection)
            data_collection = TagSubset(data_collection, self._train_test_splits)
            print(f"Keep {len(data_collection)}/{prev_len} based on train/test splits")
        if len(self._tags) > 0:
            prev_len = len(data_collection)
            data_collection = TagSubset(data_collection, self._tags)
            print(f"Keep {len(data_collection)}/{prev_len} based on tags")
        if len(self._category_tags) > 0:
            prev_len = len(data_collection)
            data_collection = CategorySubset(data_collection, self._category_tags)
            print(f"Keep {len(data_collection)}/{prev_len} based on category tags")
        if self._percentage < 1.0:
            data_collection = RandomSubset(data_collection, self._percentage)
            print(f"Keep {self._percentage * 100}% of data collection")
        return data_collection
