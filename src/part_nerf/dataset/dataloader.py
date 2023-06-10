from torch.utils.data import DataLoader

from .dataset import DatasetUnion


def build_dataloader(
    dataset: DatasetUnion,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool = True,
) -> DataLoader:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    return dataloader
