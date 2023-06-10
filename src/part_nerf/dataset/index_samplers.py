from typing import Tuple

import numpy as np


def get_all_indices(H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get all indices of the image.

    Args:
        H (int): The height of the image.
        W (int): The width of the image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: All the image indices, along with 1 valued weights.
    """
    idxs_rows = np.repeat(np.arange(H), W, axis=0)
    idxs_cols = np.tile(np.arange(W), H)
    weights = np.ones((len(idxs_rows), 1), dtype=np.float32)
    return idxs_rows, idxs_cols, weights


def get_all_positive_indices(
    H: int, W: int, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get all positive indices of the image, defined by the input mask.

    Args:
        H (int): The height of the image.
        W (int): The width of the image.
        mask (np.ndarray): The mask of the image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: All the positive image indices, along with 1 valued weights.
    """
    idxs_rows = np.repeat(np.arange(H), W, axis=0)
    idxs_cols = np.tile(np.arange(W), H)
    positive_labels = mask > 0
    idxs_rows = idxs_rows[positive_labels]
    idxs_cols = idxs_cols[positive_labels]
    weights = np.ones((len(idxs_rows), 1), dtype=np.float32)
    return idxs_rows, idxs_cols, weights


def get_uniform_indices(
    N: int, H: int, W: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get N uniformly sampled indices of the image.

    Args:
        N (int): The number of indices to sample.
        H (int): The height of the image.
        W (int): The width of the image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The sampled indices, along with 1 valued weights.
    """
    idxs_rows = np.random.choice(H, N)
    idxs_cols = np.random.choice(W, N)
    weights = np.ones((len(idxs_rows), 1), dtype=np.float32)
    return idxs_rows, idxs_cols, weights


def get_equal_indices(
    N: int, H: int, W: int, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get N indices of the image, with equal probability of sampling a positive
    and negative pixel, defined by the input mask.

    Args:
        N (int): The number of indices to sample.
        H (int): The height of the image.
        W (int): The width of the image.
        mask (np.ndarray): The mask of the image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The sampled indices, and weights defined by the
            sampled probability distribution.
    """
    n_positive = mask.sum()
    n_negative = H * W - n_positive
    p = n_negative * mask + n_positive * (1.0 - mask)
    p /= p.sum()
    # Take all rows and columns
    idxs_rows = np.repeat(np.arange(H), W, axis=0)
    idxs_cols = np.tile(np.arange(W), H)

    selected_idxs = np.random.choice(len(idxs_rows), N, p=p)
    idxs_rows = idxs_rows[selected_idxs]
    idxs_cols = idxs_cols[selected_idxs]
    weights = np.float32(1.0 / len(idxs_rows)) / p[selected_idxs]
    return idxs_rows, idxs_cols, weights.reshape((len(idxs_rows), 1))
