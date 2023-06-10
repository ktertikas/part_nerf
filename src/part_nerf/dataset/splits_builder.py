from typing import Dict, List

import pandas as pd


class SplitsBuilder:
    def __init__(self, train_test_splits_file):
        self._train_test_splits_file = train_test_splits_file
        self._splits = {}

    def train_split(self) -> List[str]:
        return self._splits["train"]

    def test_split(self) -> List[str]:
        return self._splits["test"]

    def val_split(self) -> List[str]:
        return self._splits["val"]

    def _parse_train_test_splits_file(self) -> pd.DataFrame:
        data = pd.read_csv(self._train_test_splits_file, dtype=str)
        return data

    def _parse_split_file(self) -> Dict[str, List[str]]:
        raise NotImplementedError()

    def get_splits(self, keep_splits: List[str] = ["train, val"]) -> List[str]:
        if not isinstance(keep_splits, list):
            keep_splits = [keep_splits]
        # Return only the split
        s = []
        split_file_dict = self._parse_split_file()
        for ks in keep_splits:
            s.extend(split_file_dict[ks])
        return s


class CSVSplitsBuilder(SplitsBuilder):
    def _parse_split_file(self):
        if not self._splits:
            data = self._parse_train_test_splits_file()
            split_col = data.columns[-1]
            for s in ["train", "test", "val"]:
                # Only keep the data for the current split
                d = data[data[split_col] == s]
                tags_df = d[d.columns[0]] + ":" + d[d.columns[1]]
                self._splits[s] = tags_df.to_list()

        return self._splits


def splits_builder_factory(collection_type: str) -> SplitsBuilder:
    return {
        "lego_dataset": CSVSplitsBuilder,
        "shapenet": CSVSplitsBuilder,
    }[collection_type]
