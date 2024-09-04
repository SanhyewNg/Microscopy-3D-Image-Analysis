import copy
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imageio import get_reader
from keras.preprocessing.image import array_to_img

import clb.denoising.denoise as denoise
import clb.image_processing as img_processing
from clb.dataprep import augmentations2D
from clb.dataprep.utils import (extract_label_edges, fix_data_dims,
                                get_tiff_paths_from_directories,
                                load_tiff_stack, rescale_to_float)
from clb.predict.predict_tile import load_model_with_cache
from clb.volume_slicer import split_volume
from vendor.genny.genny.wrappers import gen_wrapper


def _map_readers_to_pages(img_files, gt_files):
    """Map readers to lists of pages that can be read with them.

    Args:
        img_files (list): Paths to files with images.
        gt_files (list): Paths to files with gt.

    Returns:
        dict: Mapping readers to lists of pages.
    """
    # Create set of readers.
    readers = {(get_reader(img_file), get_reader(gt_file))
               for img_file, gt_file in zip(img_files, gt_files)}

    # Create lists of pages.
    readers_to_pages = {reader: list(range(len(reader[0])))
                        for reader in readers}

    return readers_to_pages


def _shuffle_pages(pages_lists, seed=None):
    """Shuffle lists of pages numbers in place.

    Args:
        pages_lists (Iterable): Lists of pages.
        seed (int): Random seed.
    """
    if seed is not None:
        random.seed(seed)

    for pages in pages_lists:
        random.shuffle(pages)


@gen_wrapper
def shuffled_data_generator(image_data, gt_data, seed=None, infinite=True):
    """

    Args:
        image_data (path): Directory with multipage TIFF files with images
                           or a list of directories to generate images from.
        gt_data (path): Directory with multipage TIFF files with ground truths
                        or a list of directories to generate ground truths.
        seed (int): Random seed.
        infinite (bool): Should data be generated infinitely.

    Yields:
        tuple: Image, ground truth.
    """
    if seed is not None:
        random.seed(seed)

    # Get list of paths.
    img_files = get_tiff_paths_from_directories(image_data)
    gt_files = get_tiff_paths_from_directories(gt_data)

    while True:
        readers_to_pages = _map_readers_to_pages(img_files, gt_files)

        _shuffle_pages(readers_to_pages.values(), seed)

        while readers_to_pages:
            # Draw a reader and page.
            readers, = random.sample(readers_to_pages.keys(), 1)
            page = readers_to_pages[readers].pop()

            if not readers_to_pages[readers]:
                del readers_to_pages[readers]

            # Read pages.
            img_reader, gt_reader = readers
            img, gt = img_reader.get_data(page), gt_reader.get_data(page)

            yield img, gt

        if not infinite:
            break


@gen_wrapper
def raw_data_generator(image_data, gt_data, channels=1, spatial_context=False,
                       infinite=True):
    """Generate raw data from directories.

    Args:
        image_data (path): directory with multipage TIFF files with images
                    or a list of directories to generate images from
        gt_data (path): directory with multipage TIFF files with ground truths
                    or a list of directories to generate ground truths from
        channels (int): number of channels made from adjacent images on Z axis,
                    that the input image is going to have.
                    If == 1, it's a single-channel image.
        spatial_context (bool): if True: adjust images and ground truths for
            spatial context.
        infinite (bool): loop infinitely

    Yields:
        tuple (image, ground truth)
    """

    img_files = get_tiff_paths_from_directories(image_data)
    gt_files = get_tiff_paths_from_directories(gt_data)

    if len(img_files) != len(gt_files):
        raise Exception('Number of images does not match number of labels.')

    while True:

        for img_file, gt_file in zip(img_files, gt_files):

            img_stack = load_tiff_stack(img_file)
            gt_stack = load_tiff_stack(gt_file)
            img_stack_sliced, gt_stack_sliced = split_volume(img_stack,
                                                             gt_stack,
                                                             channels,
                                                             spatial_context=
                                                             spatial_context)
            for img, gt in zip(img_stack_sliced, gt_stack_sliced):
                yield (img, gt)

        if not infinite:
            break


@gen_wrapper
def blobs_removal(data_gen, remove_blobs=False, blob_marker=1):
    """Remove areas annotated as 'unable to annotate'.

    It has to be done in order to prevent the network from being punished
    for trying to predict in the areas, where our annotator failed.

    Args:
        data_gen: input data generator. It should yield a tuple (img, gt),
                  where img is an image to remove the blob area from and the
                  gt is the ground-truth to remove the blob area from.
        remove_blobs (bool): if True, remove blobs.
        blob_marker: value of pixels marked as 'blob'

    Yields:
        tuple (image, ground truth) with blob areas removed (converted into
        black pixels both in the image and in the ground truth)
    """
    for img, gt in data_gen:
        if remove_blobs:
            no_blob_gt = gt != blob_marker
            img = np.multiply(img, no_blob_gt)
            gt = np.multiply(gt, no_blob_gt)

        yield (img, gt)


@gen_wrapper
def add_bin_channel(data_gen, obj_value=None):
    """Add binarized ground truth as new channel of `gt`.

    Args:
        data_gen: input data generator. It should yield a tuple (img, gt),
                   where img is an image (will be just yielded unmodified)
                   and gt is the ground-truth that will be binarized.
        obj_value: value to be used when pixel belongs to the object. If no
                   value is specified, than by default maximum value of
                   ground-truth datatype is taken as `obj_value`.

    Yields:
        tuple (unmodified image, ground-truth with new channel with
        binarized ground truth added).
    """
    for img, gt in data_gen:
        if len(gt.shape) != 2:
            raise ValueError('Invalid ground truth image shape: should be'
                             ' (height, width).')

        if obj_value is None:
            obj_value = np.iinfo(gt.dtype).max

        bin_channel = (obj_value * (gt > 0)).astype(gt.dtype)
        yield (img, np.dstack((gt, bin_channel)))
        obj_value = None


@gen_wrapper
def add_boundaries_channel(data_gen, boundary_thickness=2, bnd_value=None,
                           touching_boundaries=False):
    """Add binary boundaries information to the existing `gt` image.

    Note, that if `gt` is a multi-channel image, only first one will be used
    to generate boundaries.

    Args:
        data_gen: input data generator. It should yield a tuple (img, gt),
                  where img is an image (will be just yielded unmodified)
                  and gt is the ground-truth that will be binarized. For
                  best results, first channel of `gt` should contain unique
                  labels for each instance of the objects. It will work
                  with binarized mask, but the results will probably be far
                  from being acceptable.
        boundary_thickness (int): how thick is the boundary (by default it's
                                  2 pixels thick). It should always be an
                                  even number.
        bnd_value: value to be used when a pixel is a boundary. If no
                   value is specified, than by default maximum value of
                   ground-truth datatype is taken as `bnd_value`.
        touching_boundaries (bool): if True results in binary masks only with
                                    boundaries that are touching.

    Yields:
        tuple (unmodified image, ground-truth with new channel with binary
        boundaries information added).
    """
    for img, gt in data_gen:
        if len(gt.shape) < 2:
            raise (ValueError('Ground truth should be at least 2D.'))
        elif len(gt.shape) == 2:
            boundaries = \
                extract_label_edges(labels=gt,
                                    boundary_thickness=boundary_thickness,
                                    touching_boundaries=touching_boundaries)
        elif len(gt.shape) > 2:
            boundaries = \
                extract_label_edges(labels=gt[..., 0],
                                    boundary_thickness=boundary_thickness,
                                    touching_boundaries=touching_boundaries)

        if bnd_value is None:
            bnd_value = np.iinfo(gt.dtype).max

        boundaries = boundaries.astype(gt.dtype)
        boundaries = bnd_value * boundaries

        yield (img, np.dstack((gt, boundaries)))
        bnd_value = None


@gen_wrapper
def subtract_boundaries_from_objects(data_gen, obj_channel=1, bnd_channel=2):
    """Uses boundaries channel to subtract it from objects channel.

    Main use case is there are two binary channels available: boundaries and
    objects. To separate the objects (e.g. when they're touching), it would be
    good to subtract the boundaries from objects channel.

    Args:
        data_gen: input data generator. It should yield a tuple (img, gt),
                  where img is an image (will be just yielded unmodified)
                  and gt is the multi-channel ground-truth. Expected
                  channels (at least): 1st channel - original ground-truth,
                  2nd channel - binary objects channel, 3rd channel - binary
                  boundaries channel. 2nd and 3rd channel will be used in
                  this function.
        obj_channel: channel with objects (counted from 0)
        bnd_channel: channel with boundaries (counted from 0)

    Yields:
        tuple (unmodified image, modified `gt` object - it's 2nd channel
        with binary objects will be merged with the information from 3rd
        channel (binary boundaries channel).

    Raises:
        ValuError when there are less then 3 channels in `gt` array.
    """
    for img, gt in data_gen:
        mod_gt = copy.copy(gt)
        objects = gt[..., obj_channel]
        boundaries = gt[..., bnd_channel]
        mod_gt[..., obj_channel] = objects * (boundaries == 0)
        yield (img, mod_gt)


@gen_wrapper
def normalizer(data_gen):
    """Normalize data from `data_gen`.

    Args:
        data_gen: input data generator. It should yield a tuple (img, gt).

    Yields:
        tuple (normalized img, normalized_gt).
    """
    for img, gt in data_gen:
        norm_img = rescale_to_float(img)
        norm_gt = gt

        for channel in range(gt.shape[-1]):
            channel_data = gt[..., channel]
            unique_val = np.unique(channel_data)
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)

            if len(unique_val) == 2 and min_val == 0 and max_val > 1:
                norm_gt[..., channel] = np.divide(channel_data,
                                                  max_val).astype(gt.dtype)

        yield (norm_img, norm_gt)


@gen_wrapper
def rescaler(data_gen, trim_method, out_dim):
    """Rescale data from `data_gen`.

    Args:
        data_gen: input data generator. It should yield a tuple (img, gt).
        trim_method: method to fix image dimensions
        out_dim: size of the image that enters the network
    Yields:
        tuple (rescaled_img, rescaled_gt).

    Raises:
        NotImplementedError when trim_method == resize.
    """
    for img, gt in data_gen:
        rescaled_img, rescaled_gt = fix_data_dims(images=[img],
                                                  masks=[gt],
                                                  trim_method=trim_method,
                                                  out_dim=out_dim)

        if trim_method == 'resize':
            warnings.warn('Resizing may spoil ground-truths binary nature.')

        yield (rescaled_img[0], rescaled_gt[0])


@gen_wrapper
def form_tensors(data_gen):
    """Form tensors from data that will match Keras requirements.

    Args:
        data_gen: input data generator. It is expected to yield tuples
                  (img, gt)

    Yields:
        tuple : (
                    image tensor - numpy array: (1, height, width, channels),
                    gt tensor: (1, height, width, channels)
                )

    """
    for img, gt in data_gen:
        img_tensor = img
        gt_tensor = gt

        # Expand if there is no channel info
        if len(img.shape) == 2:
            img_tensor = np.expand_dims(img_tensor, axis=-1)
        if len(gt.shape) == 2:
            gt_tensor = np.expand_dims(gt_tensor, axis=-1)

        img_tensor = np.expand_dims(img_tensor, axis=0)
        gt_tensor = np.expand_dims(gt_tensor, axis=0)

        yield (img_tensor, gt_tensor)


@gen_wrapper
def batch_generator(data_gen, batch_size=1):
    """Form batches from images and ground-truths.

    Args:
        data_gen: generator that will provide data for batching in a tuple:
                  (4D image tensor, 4D ground-truth tensor)
        batch_size: size of the output batch

    Returns:
        tuple (
                4D tensor of shape [images, height, width, 1],
                4D tensor of shape [ground-truths, height, width, channels]
              )
    """
    batch_empty = True
    for img, gt in data_gen:
        if batch_empty:
            img_batch = img
            gt_batch = gt
            batch_empty = False
        else:
            img_batch = np.append(img_batch, img, axis=0)
            gt_batch = np.append(gt_batch, gt, axis=0)

        if img_batch.shape[0] == batch_size:
            yield (img_batch, gt_batch)
            batch_empty = True


@gen_wrapper
def single_image_generator(data_gen, multiple_gt_outputs=False):
    """Generate single images from batched data.

    Args:
        data_gen: generator yielding data in batches. First dimensions is
                  assumed to represent number of images in a single batch:
                  (num_images, height, width, [optional] channels).
        multiple_gt_outputs: if trained model has multiple outputs as a
                             network (e.g. DCAN has objects and boundaries
                             outputs) they are packed in a list of 4D numpy
                             arrays. Thus, it's necessary to unbatch them in
                             a slightly different way, than for standard
                             single output models.

    Yields:
        tuple: (image, ground truth)
    """
    for img_batch, gt_batch in data_gen:

        if not multiple_gt_outputs:
            for example in zip(img_batch, gt_batch):
                yield example
        else:

            if not isinstance(gt_batch, list):
                raise TypeError('When multiple_gt_outputs is set to True, '
                                'gt_batch should be a list, but is '
                                '{}'.format(type(gt_batch)))

            for n in range(img_batch.shape[0]):
                # np.newaxis keeps 4D nature of the output.
                yield np.array(img_batch[np.newaxis, n]), \
                      [output[np.newaxis, n] for output in gt_batch]


@gen_wrapper
def form_dcan_input(data_gen, obj_channel=1, bnd_channel=2):
    """Form batched data from `data_gen` into the format suitable for DCAN.

    Args:
        data_gen: generator that will provide batched data in a tuple:
                  (4D images tensor, 4D ground-truths tensor). 4D
                  ground-truths tensor requires at least 2 channels: 1 with
                  objects and 1 with boundaries. Default values for function
                  parameters assume these channels' locations.
        obj_channel: number of channel with objects (counted from 0)
        bnd_channel: number of channel with boundaries (counted from 0)

    Yields:
        tuple (
            4D images tensor, [4D objects tensor, 4D boundaries tensor]
        )
    """
    for img_batch, gt_batch in data_gen:
        yield (img_batch, [gt_batch[..., obj_channel, np.newaxis],
                           gt_batch[..., bnd_channel, np.newaxis]])


@gen_wrapper
def three_class_gt(data_gen, bin_object_channel, bin_boundary_channel):
    """Generator that converts ground truth into 3-class RGB label.

    Args:
        data_gen: Generator yielding single pairs of (img, gt). Both img and
                  gt has to be a 3D tensors of shape (height, width,
                  channels). `img` can be anything (it's not analyzed in
                  this function), but `gt` must be a grayscale image with
                  each separate object marked with unique integer value (all
                  the pixels belonging to a specific object should have
                  identical values).
        bin_object_channel (int): number of a channel with binary objects
        bin_boundary_channel (int): number of a channel with binary boundaries
    Yields:
        tuple (
            3D image tensor (height, width, channels)
            3D ground truth tensor (height, width, 3)
        )
    """
    for img, gt in data_gen:
        objects = gt[..., bin_object_channel]
        boundaries = gt[..., bin_boundary_channel]
        max_value = np.iinfo(gt.dtype).max

        # BINARY LABEL

        # prepare buffer for binary label
        label_binary = np.zeros((gt.shape[:2] + (3,)), dtype=gt.dtype)

        # write binary label
        label_binary[(objects == 0) & (boundaries == 0), 0] = max_value
        label_binary[(objects != 0) & (boundaries == 0), 1] = max_value
        label_binary[boundaries == max_value, 2] = max_value

        yield img, label_binary


@gen_wrapper
def denoising(data_gen, model_path='models/denoising/model0.h5'):
    """Denoising preprocess.

    Args:
        data_gen: Generator yielding pairs image, ground truth.
        model_path (str): Path to denoising model.

    Yields:
        tuple: Image after denoising, ground truth.
    """
    model = load_model_with_cache(model_path)

    for img, gt in data_gen:
        denoised_img = denoise.denoise_image(image=img,
                                             model=model,
                                             batch_size=1,
                                             patches_shape=None,
                                             patches_stride=None)

        yield denoised_img, gt


@gen_wrapper
def clahe(data_gen):
    """Clahe preprocess.

    Args:
        data_gen: Generator yielding pairs image, ground truth.

    Yields:
        tuple: Image after clahe, ground truth.
    """
    for img, gt in data_gen:
        img_rescaled = rescale_to_float(img, float_type='float32')
        yield img_processing.clahe(img_rescaled, size=70, median_size=2), gt


class UnrecognizedPreprocessingError(Exception):
    """Raised when trying to use unknown preprocessing."""


@gen_wrapper
def preprocess(data_gen, preprocessings):
    """Apply `preprocessings to each element from `data_gen`.

    Args:
        data_gen: Generator yielding pairs image, ground truth.
        preprocessings (iterable): Preprocessings to apply. Currently supported:
                                   - denoising
                                   - clahe

    """
    for preprocessing in preprocessings:
        if preprocessing == 'denoising':
            data_gen = data_gen | denoising()
        elif preprocessing == 'clahe':
            data_gen = data_gen | clahe()
        else:
            raise UnrecognizedPreprocessingError('Unrecognized preprocessing: {}'
                                                 .format(preprocessing))

    yield from data_gen


def dcan_dataset_generator(image_data, gt_data, batch_size, out_dim, augs,
                           trim_method, seed, boundary_thickness=2,
                           enable_elastic=True, remove_blobs=False,
                           touching_boundaries=False, infinite=True, preprocessings=()):
    """Generator yielding data ready for training DCAN network.

       It should be used directly with `model.fit_generator()` function (or sth
       similar). `dcan_dataset_generator()` is a generator that unifies all
       other generators - loading raw data, batching images, preprocessing,
       filtering and augmenting.

    Args:
        image_data: directory with multipage TIFF files with images or a list
                     of directories to generate images from
        gt_data: directory with multipage TIFF files with ground truths or a
                  list of directories to generate ground truths from
        batch_size: size of the batch
        out_dim: size of the image that enters the network
        augs: number of augmented images that will be produces additionally
        trim_method: method to fix image dimensions
        seed: random seed used in augmentator to generate exact augmentations
        boundary_thickness (int): how thick is the boundary (by default it's
                                  2 pixels thick). It should always be an
                                  even number.
        enable_elastic (bool): should elastic distortions be applied as part
                               of augmentations.
        remove_blobs (bool): if True, removing blobs.
        touching_boundaries (bool): if True results in binary masks only with
                                    boundaries that are touching.
        infinite: should data be generated infinitely
        preprocessings (iterable): What preprocessings to use, currently supported:
                                   - denoising
                                   - clahe

    Yields:
        tuple: (
                    numpy array with input images (num_imgs, height, width, 1),
                    list: [4D array with objects, 4D array with boundaries]
               )

        Such output format is required to train DCAN network with 2 outputs
        (1 for objects and 1 for boundaries).

    """
    if trim_method == 'padding' or trim_method == 'resize':
        mode = 'constant'
        pad = None
    elif trim_method == 'reflect':
        mode = 'reflect'
        pad = out_dim  # Pad is equal to the length of image's edge.

    augmentator = augmentations2D.AugGenerator(pad=pad, mode=mode,
                                               enable_elastic=enable_elastic,
                                               seed=seed)

    gen = (raw_data_generator(image_data=image_data, gt_data=gt_data,
                              infinite=infinite) |
           blobs_removal(remove_blobs=remove_blobs) |
           add_bin_channel() |
           add_boundaries_channel(boundary_thickness=boundary_thickness,
                                  touching_boundaries=touching_boundaries) |
           preprocess(preprocessings=preprocessings) |
           subtract_boundaries_from_objects() |
           augmentator.flow(augs=augs) |
           normalizer() |
           rescaler(trim_method=trim_method, out_dim=out_dim) |
           form_tensors() |
           batch_generator(batch_size=batch_size) |
           form_dcan_input())

    return gen


def unet_dataset_generator(image_data, gt_data, channels, batch_size,
                           out_dim, augs, trim_method, seed, enable_elastic,
                           boundary_thickness=2, remove_blobs=False,
                           touching_boundaries=False, infinite=True):
    """Generator yielding data ready for training multiclass U-net network.

       It should be used directly with `model.fit_generator()` function (or sth
       similar). `unet_dataset_generator()` is a generator that unifies all
       other generators - loading raw data, batching images, preprocessing,
       filtering and augmenting.

    Args:
        image_data: directory with multipage TIFF files with images or a list
                 of directories to generate images from
        gt_data: directory with multipage TIFF files with ground truths or a
                 list of directories to generate ground truths from
        channels: number of channels made from adjacent images on Z axis,
                 that the input image is going to have.
                 If == 1, it's a single-channel image.
        batch_size: size of the batch
        out_dim: size of the image that enters the network
        augs: number of augmented images that will be produces additionally
        trim_method: method to fix image dimensions
        seed: random seed used in augmentator to generate exact augmentations
        boundary_thickness (int): how thick is the boundary (by default it's
                                  2 pixels thick). It should always be an
                                  even number.
        enable_elastic (bool): should elastic distortions be applied as part
                               of augmentations.
        remove_blobs (bool): if True, removing blobs.
        touching_boundaries (bool): if True results in binary masks only with
                                    boundaries that are touching.
        infinite: should data be generated infinitely

    Yields:
        tuple: (
                    numpy array with input images (num_imgs, height, width, 1),
                    numpy array with ground truths (num_imgs, height, width, 3),
               )

        Such output format is required to train U-net network with single
        3-channel output (first channel for background, second for objects'
        interiors and the third one for boundaries).

    """
    if trim_method == 'padding' or trim_method == 'resize':
        mode = 'constant'
        pad = None
    elif trim_method == 'reflect':
        mode = 'reflect'
        pad = out_dim  # Pad is equal to the length of image's edge.

    augmentator = augmentations2D.AugGenerator(pad=pad, mode=mode,
                                               enable_elastic=enable_elastic,
                                               seed=seed)

    gen = (raw_data_generator(image_data=image_data, gt_data=gt_data,
                              channels=channels, spatial_context=False,
                              infinite=infinite) |
           blobs_removal(remove_blobs=remove_blobs) |
           add_bin_channel() |
           add_boundaries_channel(boundary_thickness=boundary_thickness,
                                  touching_boundaries=touching_boundaries) |
           three_class_gt(bin_object_channel=1, bin_boundary_channel=2) |
           augmentator.flow(augs=augs) |
           normalizer() |
           rescaler(trim_method=trim_method, out_dim=out_dim) |
           form_tensors() |
           batch_generator(batch_size=batch_size))

    return gen


def main():
    image_data = 'data/instance/training/T8/train/images'
    gt_data = 'data/instance/training/T8/train/labels'
    channels = 3
    batch_size = 4
    out_dim = 256
    trim_method = 'resize'
    augs = 2
    seed = 43

    data_gen = unet_dataset_generator(image_data=image_data,
                                      gt_data=gt_data,
                                      channels=channels,
                                      batch_size=batch_size,
                                      out_dim=out_dim,
                                      trim_method=trim_method,
                                      augs=augs,
                                      seed=seed,
                                      infinite=True)

    for imgs, gts in data_gen:
        for idx in range(imgs.shape[0]):
            plt.subplot(1, 3, 1)
            plt.imshow(array_to_img(imgs[idx]), cmap='gray')
            plt.subplot(1, 3, 2)
            plt.imshow(array_to_img(gts[idx]))
            plt.suptitle('batch part {}/{}'.format(idx + 1, len(imgs)))
            plt.show()


if __name__ == '__main__':
    main()
