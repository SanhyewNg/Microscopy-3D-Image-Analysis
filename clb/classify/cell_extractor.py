import math
import os

import numpy as np
from skimage import measure, transform
from tqdm import tqdm

from clb.classify.utils import add_data_with_prefix
from clb.cropping.volumeroi import VolumeROI
from clb.image_processing import resample
from clb.utils import replace_values, channels_count, spatial_shape

from clb.classify.extractors import DESIRED_VOXEL_SIZE, DESIRED_VOXEL_UM, preprocess_input_labels, \
    extract_channels
from clb.cropping import CropInfo


def extract_cells_crops(images_volume, labels_volume, crop_size=12, voxel_size=DESIRED_VOXEL_SIZE):
    """
    Extract cell morphological and channel related features.
    Args:
        images_volume: S x Y x X x C or S x Y x X volume for which we want to calculate features
            assumes that it is already scaled to 0-1 range
        labels_volume: S x Y x X
            cell level segmentation as cell labels
        crop_size: side of the cube around each cell in um
        voxel_size: tuple with the real world size of the voxel (z,y,x), especially important to normalize Z axis which usually
            has much lower resolution (voxel is large in Z axis)
    Returns:
        dictionary of cells to cell crops in form of numpy volumes created after resizing to DESIRED_VOXEL_SIZE
    """
    assert spatial_shape(images_volume) == spatial_shape(labels_volume), \
        "Volumes shapes do not match: image_volume={0}, labels_volume={1}".format(spatial_shape(images_volume),
                                                                                  spatial_shape(labels_volume))

    res = {}

    pixel_crop_shape = np.ceil(crop_size / np.array(voxel_size)).astype(np.int32)
    expected_shape = np.ceil(crop_size / np.array(DESIRED_VOXEL_SIZE)).astype(np.int32)

    image_channels_count = channels_count(images_volume)
    expected_shape_with_channels = expected_shape.copy()
    if image_channels_count is not None:
        expected_shape_with_channels = np.append(expected_shape_with_channels, image_channels_count)

    props = measure.regionprops(label_image=labels_volume)
    for prop in tqdm(props, mininterval=2):
        i = prop.label

        my_crops = {}
        cell_crop_infos = CropInfo.create_centered_volume(prop.centroid, pixel_crop_shape)

        cell_roi = VolumeROI.create_empty(cell_crop_infos,
                                          dtype=images_volume.dtype, channels=image_channels_count)
        bounded_roi = VolumeROI.from_absolute_crop(cell_crop_infos, images_volume)

        labels_roi = VolumeROI.create_empty(cell_crop_infos,
                                            dtype=labels_volume.dtype)
        bounded_labels_roi = VolumeROI.from_absolute_crop(cell_crop_infos, labels_volume)

        cell_roi.implant(bounded_roi)
        labels_roi.implant(bounded_labels_roi)

        # rescale cell to standard size
        resampled_cell_image, _ = resample(cell_roi.crop_volume, voxel_size, DESIRED_VOXEL_SIZE)
        resampled_cell_label, _ = resample(labels_roi.crop_volume, voxel_size, DESIRED_VOXEL_SIZE, order=0)

        resized_cell_image = transform.resize(resampled_cell_image, expected_shape_with_channels,
                                              order=1, preserve_range=True, anti_aliasing=False,
                                              mode='constant').astype(images_volume.dtype)
        resized_cell_label = transform.resize(resampled_cell_label, expected_shape,
                                              order=0, preserve_range=True, anti_aliasing=False,
                                              mode='constant').astype(labels_volume.dtype)

        if prop.label not in np.unique(resized_cell_label):
            print("Cell {} of pixel size {} not visible in crop after rescaling.".format(prop.label, prop.area))
        assert resized_cell_image.shape[:3] == tuple(expected_shape)
        assert resized_cell_label.shape == tuple(expected_shape)

        my_crops['input'] = resized_cell_image
        my_crops['contour'] = resized_cell_label

        my_crops['id'] = i
        res[i] = my_crops

    return res


def extract_all_cells_crops(images_volume, labels_volume, channels_with_preprocessing_list,
                            crop_size,
                            only_for_labels=None,
                            voxel_size=DESIRED_VOXEL_SIZE):
    """
    Extract crop for each cell with requested channels / preprocessing.
    Args:
        images_volume: S x Y x X x C or S x Y x X
            with channel for which we want to calculate features
        labels_volume: S x Y x X
            cell level segmentation as cell labels
        channels_with_preprocessing_list: list of channels with optional preprocessings to use e.g. ["1","2"] or ["1-equal","1"]
        crop_size: side of the cube around each cell in um
        only_for_labels: list of label value for which to calculate features
            if None then all are calculated
        voxel_size: tuple with the real world size of the voxel (z,y,x), especially important to normalize Z axis which usually
            has much lower resolution (voxel is large in Z axis)
            if None then no voxel resizing should be done

    Returns:
        dictionary of cells to cell crops in form of numpy volumes created after resizing to DESIRED_VOXEL_SIZE
    """
    voxel_size = voxel_size or DESIRED_VOXEL_SIZE

    if only_for_labels is not None:
        labels_volume = replace_values(labels_volume, {i: i for i in only_for_labels}, zero_unmapped=True)

    res = {}

    # TODO add preprocessings
    # Rescale and pick only important channels.
    # for channel_with_preprocess in tqdm(channels_with_preprocessing_list):
    #     channel_volume, preprocess_labels_volume = preprocess_input_labels(images_volume, labels_volume, channel_with_preprocess, voxel_size)
    #
    #     channel_crops = extract_cells_crops(channel_volume, preprocess_labels_volume, crop_size, voxel_size)
    #     add_data_with_prefix(res, channel_crops, str(channel_with_preprocess))
    channels = extract_channels(channels_with_preprocessing_list)
    channels_volume = images_volume[..., channels]

    channels_crops = extract_cells_crops(channels_volume, labels_volume, crop_size, voxel_size)
    add_data_with_prefix(res, channels_crops, "raw")

    return res
