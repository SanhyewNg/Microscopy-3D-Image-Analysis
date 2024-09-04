"""Module defines postprocess tools for denoising.

Here postprocess is understood as everything that happens between getting
network output and saving it.
"""
import itertools as it

import numpy as np


class WrongPatchSize(Exception):
    """Raised when patches don't fill whole image."""


def merge_patches(patches, image_shape, stride):
    """Merge image divided to overlapping patches.

    Overlapping areas are averaged.

    Args:
        patches (np.ndarray): Patches, shape (N, y, x).
        image_shape (tuple): Shape of output image, (Y, X).
        stride (tuple): Stride used during patch extraction.

    Returns:
        np.ndarray: Reconstructed image, shape (Y, X).
    """
    image = np.zeros(image_shape)
    overlap_counts = np.zeros(image_shape)
    rows = range(0, image_shape[0] - patches.shape[1] + 1, stride[0])
    cols = range(0, image_shape[1] - patches.shape[2] + 1, stride[1])

    for patch, (row, col) in zip(patches, it.product(rows, cols)):
        end_row = row + patch.shape[0]
        end_col = col + patch.shape[1]
        image[row:end_row, col:end_col] += patch
        overlap_counts[row:end_row, col:end_col] += 1

    if (overlap_counts == 0).any():
        raise WrongPatchSize('Patches do not fill whole image.')

    return image / overlap_counts
