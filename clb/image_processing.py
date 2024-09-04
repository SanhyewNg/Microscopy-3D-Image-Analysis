import itertools
import logging
from collections import Counter

import daiquiri
import imageio
import numpy as np
import scipy.ndimage
import skimage.exposure
import skimage.filters.rank
import skimage.morphology
import skimage.segmentation
from scipy.signal import fftconvolve
from skimage.morphology import disk

from clb.dataprep.utils import extract_label_edges
from clb.utils import has_gaps


daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


def resample(volume, pixel_size=(1, 1, 1), new_pixel_size=(1, 1, 1), tolerance=0.1, order=1):
    """
    Resample the data according to the nex pixel size.
    If both pixel sizes are the same then volume is returned.
    Args:
        volume: S x Y x X numpy volume
        pixel_size: pixel size of the input volume
        new_pixel_size: desired new pixel size
        tolerance: tolerance to pixel size difference
        order: order of the spline interpolation

    Returns:
        resampled_volume: S` x Y` x X` with pixel size close to new_pixel_size
        final_pixel_size: actual pixel size of the resampled_volume (should be very close to new_pixel_size
                        or None if resampling was not required
    """
    assert not volume.dtype == np.bool, "Bool arrays are not supported."
    if volume.size == 1:
        print("Volume of size 1 was given to resampling.")
        return volume, None

    if not all(np.isclose(pixel_size, new_pixel_size, rtol=0, atol=tolerance)):
        pixel_size = np.array(pixel_size, dtype=np.float32)
        resize_factor = pixel_size / new_pixel_size

        new_real_shape = volume.shape[:3] * resize_factor
        # make sure that even if down sample it will never get reduced to single value
        new_shape = np.maximum(2, np.round(new_real_shape))
        real_resize_factor = new_shape / volume.shape[:3]

        final_pixel_size = pixel_size / real_resize_factor

        if volume.ndim == 4:
            rescaled = []
            for i in range(volume.shape[-1]):
                rescaled.append(
                    scipy.ndimage.interpolation.zoom(volume[..., i], real_resize_factor, mode='nearest', order=order))
            resampled_volume = np.stack(rescaled, axis=3)
        else:
            resampled_volume = scipy.ndimage.interpolation.zoom(volume, real_resize_factor, mode='nearest', order=order)

        if not np.any(resampled_volume):
            print("No positive pixels after resampling.")
            return volume, None

        return resampled_volume, final_pixel_size

    return volume, None


def correlate2d_with_fft(in1, in2):
    return fftconvolve(in1, in2[::-1, ::-1], mode='same')


def find_corresponding_labels(map_from, map_to, return_overlap=False, return_count=False):
    """
    Find mapping from map_from values to map_to.

    Args:
        map_from: numpy array for first label image
        map_to: numpy array for next label image
        return_overlap: add information about the overlap fraction to
                        resulting mapping
        return_count: add information about the overlap pixels to
                        resulting mapping
    Returns:
        dictionary with map_from to map_to. If return_overlap or return_count then tuple is
        returned with overlap_fraction, return_count.
    """
    overlap = (map_from > 0) & (map_to > 0)
    values_labels1 = map_from[overlap]
    values_labels2 = map_to[overlap]

    # count pixels in map_from
    map_from_counts = Counter(map_from.flatten())

    # count pixels that overlap in map_from and map_to
    pairs = zip(values_labels1, values_labels2)
    c = Counter(pairs)

    # for each value in map_from choose the one in map_to with largest overlap
    by_overlap = sorted(c.items(), key=lambda x: -x[1])

    mappers = {}
    for (left, right), overlap in by_overlap:
        if left not in mappers:
            overlap_fraction = float(overlap) / map_from_counts[left]

            summary = [right]
            if return_overlap:
                summary.append(overlap_fraction)
            if return_count:
                summary.append(overlap)

            mappers[left] = summary[0] if len(summary) == 1 else tuple(summary)
    return mappers


def restrict_struct_to_only_2d(struct):
    """
    Keeps only central layer of the 3d struct
    which should correspond to no 3d morphology.
    Args:
        struct: Y x X or S x Y x X numpy array

    Returns:
        new numpy array with only central layer preserved.
    """
    if struct.ndim == 3:
        res = np.zeros(struct.shape, struct.dtype)
        center_slice = struct.shape[0] // 2
        res[center_slice] = struct[center_slice]
        return res
    else:
        return struct.copy()


def remove_annotated_blobs_when_overlap(label_image, label_with_blobs_image, blob_value=1, minimum_blob_overlap=0.0):
    """
    Remove any labels that overlap mostly with blob area.

    Args:
        label_image: two-dim numpy array with labels from which we are to
                     remove blob areas.
        label_with_blobs_image: two-dim numpy array with blobs marked
        blob_value: depends on label image, default is 1
        minimum_blob_overlap: minimum intersection / object to match object
                              with blob (only if blob is the most overlaping
                              label.

    Returns:
        image with blobs zeroed out.
    """
    label_image_without_blobs = label_image.copy()

    most_overlapping_values = find_corresponding_labels(
        label_image, label_with_blobs_image, return_overlap=True
    )

    for org, (label_or_blob, overlap) in most_overlapping_values.items():
        if label_or_blob == blob_value and (overlap >= minimum_blob_overlap):
            label_image_without_blobs[label_image_without_blobs == org] = 0

    return label_image_without_blobs


def remove_annotated_blobs(image, label_image, blob_value=1):
    """
    Remove blobs from given image.

    Args:
        image: two-dim numpy array to remove blob areas from.
        label_image: two-dim numpy array with blobs marked
        blob_value: depends on label image, default is 1

    Returns:
        image with blobs zeroed out.
    """

    to_clear = label_image == blob_value
    to_clear = skimage.morphology.binary_dilation(to_clear)
    image = image.copy()
    image[to_clear] = 0
    return image


def remove_gaps_in_slices(array):
    """
    Remove gaps in 3d numpy array, it modifies the input volume.
    Args:
        array: S x Y x X numpy array

    Returns:
        S x Y x X numpy array in which non-empty slices make are contiguous block
    """
    slices_structure = np.zeros((3, 3, 3), dtype=np.bool)
    slices_structure[:, 1, 1] = True
    while has_gaps(array):
        dilated = scipy.ndimage.morphology.grey_dilation(array, footprint=slices_structure)
        empty_spaces = array == 0
        np.copyto(array, dilated, where=empty_spaces)

    return array


def separate_objects(objects, boundaries, obj_thresh=0.5, bound_thresh=0.5):
    """Use boundaries information to split touching objects.

    Args:
        objects: numpy array containing mask with objects [dim_x, dim_y, 1]
        boundaries: numpy array containing mask with boundaries
                    [dim_x, dim_y, 1]
        obj_thresh: threshold to filter out objects
        bound_thresh: threshold to include boundaries information in filtering

    Returns:
        Numpy array with separated mask
    """
    output = np.zeros(objects.shape)
    output[(objects > obj_thresh) & (boundaries < bound_thresh)] = 1

    return output


def clahe(image, size, median_size=None, **kwargs):
    """
    Clahe with median filter preprocessing.
    Args:
        image: Y x X normalized to 0-1
        size: kernel_size in clahe
        median_size: disk size in median filter
        **kwargs: additional params passed to equalize_adapthist

    Returns:
        preprocessed image of the same shape as input and float32 type
    """
    assert image.dtype == np.float16 or image.dtype == np.float32 or image.dtype == np.float64
    assert np.min(image) >= 0 and np.max(image) <= 1
    assert image.ndim == 2

    if median_size is not None:
        image = skimage.img_as_float32(skimage.filters.rank.median(image, skimage.morphology.disk(median_size)))
    data_clahe = skimage.img_as_float32(skimage.exposure.equalize_adapthist(image, size, **kwargs))
    return data_clahe


def extend_membrane(image, scale=1):
    """
    Enlarge membrane-like structure using morphology.
    Args:
        image: Y x X grayscale image normalized to 0-1
        scale: optional integer scale of the enlargement

    Returns:
        processed image with membrane enlarged and float32 type
    """
    assert image.dtype == np.float16 or image.dtype == np.float32 or image.dtype == np.float64
    assert np.min(image) >= 0 and np.max(image) <= 1
    assert image.ndim == 2

    cleaned = skimage.filters.gaussian(image, scale)
    extended = skimage.morphology.dilation(cleaned, disk(scale * 2))
    smoothed = skimage.filters.gaussian(extended, scale * 2)
    reconnected = skimage.morphology.closing(smoothed, disk(scale * 4))
    return skimage.img_as_float32(reconnected)


def estimate_membrane_from_nucleus(labels, scale=1):
    """
    Estimate cells membrane from existing cell labels.
    Args:
        labels: Y x X label numpy array (uint32)
        scale: optional integer scale of the membrane size

    Returns:
        labeled membrane of the cells
    """
    assert labels.dtype == np.uint32 or labels.dtype == np.uint16 or labels.dtype == np.uint8
    assert labels.ndim == 2

    dilated = skimage.morphology.dilation(labels, disk(4 * scale))
    cell_exterior = dilated * (labels == 0)

    edges_only_small = labels * extract_label_edges(labels, 4 * scale)
    cell_exterior[edges_only_small != 0] = edges_only_small[edges_only_small != 0]

    return cell_exterior


def enhance_contour_visibility(volume, target_cell_value):
    """Enhance contour visibility of main cell by setting its pixels value to max. 
       Background pixel values are set between 64 and 164 
       
    Args:
        volume (np.array): Image to enhance (one channel)
        target_cell_value (int): Pixel value of target_cell in input volume
        
    Return:
        output (np.array): Volume where target cell has pixels values equals to 255 (uint8)
    """
    background_cell_colors = tuple(range(64, 165))
    all_colors = set(np.unique(volume))
    if target_cell_value not in all_colors:
        logger.warn("{}, {}".format(target_cell_value, all_colors))
        raise ValueError("Target cell value not found: {}".format(target_cell_value))
    cell_colors = all_colors.difference((0, target_cell_value))
    color_dist = len(background_cell_colors) // len(cell_colors) if len(cell_colors) > 1 else 1
    background_cell_colors = background_cell_colors[0::color_dist] if color_dist > 1 else background_cell_colors
    cell_color_mapping = {
        cell_value: color for (color, cell_value) in zip(itertools.cycle(background_cell_colors), cell_colors)
    }
    cell_color_mapping[target_cell_value] = 255

    output = np.zeros(volume.shape, dtype=np.uint8)
    for cell_value, target_color in cell_color_mapping.items():
        output[volume == cell_value] = target_color
    return output
