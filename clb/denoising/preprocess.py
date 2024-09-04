"""Module defines preprocess tools for denoising.

Here preprocess is understood as everything that happens between reading images
and feeding them to network, either for training or prediction. So module
includes e.g. function for augmentation as well as function for dividing image
into patches.

Attributes:
    _preprocessings_to_nouts (dict): Mapping preprocessing name to number of its outputs.
"""
import functools as ft
import itertools as it

import math
import more_itertools as mit
import numpy as np
import sklearn.feature_extraction.image as sklimage
import tensorflow as tf


_preprocessings_to_nouts = {}


def preprocess(nouts):
    """Register preprocessing in `_preprocessings_to_nouts`."""
    def wrapper(f):
        _preprocessings_to_nouts[f.__name__] = nouts
        @ft.wraps(f)
        def inner_wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return inner_wrapper

    return wrapper


@preprocess(nouts=8)
def augment(image):
    """Return augmentations of `image`.

    First image is rotated by 0, 90, 180 and 270 degrees, then image is flipped
    up to down and rotations are repeated. All received images are returned
    in form of tf.data.Dataset.

    Args:
        image (np.ndarray): Image to be augmented.

    Returns:
        list: Image with its augmentations.
    """

    rots = [np.rot90(image, k) for k in range(4)]
    flipped_image = np.flipud(image)
    rots_after_flip = [np.rot90(flipped_image, k) for k in range(4)]

    return rots + rots_after_flip


def add_noise(image, noiser=tf.random_normal, **kwargs):
    """Add noise to `image`.

    After adding noise values are clipped, so they are between 0. and 1.
    Args:
        image (tf.Tensor): Image to noise.
        noiser (Callable): Used for creating the noise.

    Returns:
        tf.Tensor: Noised image.
    """
    noise = noiser(dtype=image.dtype, shape=tf.shape(image), **kwargs)
    noised = image + noise
    clipped = tf.clip_by_value(noised, 0., 1.)

    return clipped


def extract_patches(image, shape=(128, 128), stride=(32, 32)):
    """Extract patches from `image`.

    Image is padded, so that all patches fit.

    Args:
        image (np.ndarray): 2d image.
        shape (tuple): 2d shape of one patch (y, x).
        stride (tuple): 2d shape of strides (same as above).

    Returns:
        numpy.ndarray: Extracted patches, shape (N, y, x).
    """
    patches = sklimage.extract_patches(arr=image, patch_shape=shape,
                                       extraction_step=stride)

    return patches.reshape(-1, *shape)


class Dataset:
    """Class representing a processing pipeline for the network."""
    def __init__(self, data, infinite=True):
        self._len = len(data)
        if infinite:
            self._data = it.cycle(data)
        else:
            self._data = data

    def batch(self, batch_size):
        """Create batches.

        Args:
            batch_size (int): Size of one batch, if there is a remainder last batch will
                              be smaller.
        """
        self._data = (np.stack(batch, axis=1)
                      for batch in mit.chunked(self._data, batch_size))
        self._len = math.ceil(len(self) / batch_size)

        return self

    def transform(self, func):
        """Transform each sample of data by calling `func` on it.

        If `func` was registered with `preprocess` decorator with some number of outputs
        (meaning that it's creating couple of samples from one, like augmentation),
        length of the dataset will be updated.

        Args:
            func (Callable): Transformation to apply.
        """
        nouts = _preprocessings_to_nouts.get(func.__name__, 1)
        if nouts > 1:
            self._data = it.chain.from_iterable(zip(func(img), func(gt))
                                                for img, gt in self._data)
        else:
            self._data = ((func(img), func(gt)) for img, gt in self._data)

        self._len *= nouts

        return self

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._data)

    def __len__(self):
        """Length of the dataset.

        It's always finite. If the dataset is repeated it's a length of the repeated
        part.

        Returns:
            int: Length of the dataset.
        """
        return self._len
