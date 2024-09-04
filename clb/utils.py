import cv2
import imageio
import numpy as np
import skimage.segmentation
from PIL import Image

from clb.dataprep.utils import fix_data_dims


def crop_image(full_data, crop_shape, top_left, multi_channel):
    """
    Crop imagery with rectagle shape. Works with both multi-channel and
    multi-page input.

    Args:
        full_data: NxM or NxMxC or HxNxM or HxNxMxC numpy array
        crop_shape: height, width of the crop shape
        top_left: top-left corner of the crop
        multi_channel: full_data has multiple channels

    Returns:
        NxM or NxMxC or HxNxM or HxNxMxC depending on the full_data and
        multi_channel arguments
    """
    # Ensure that slice is > 0.
    top_left = np.maximum(0, top_left)
    bottom_right = np.maximum(0, np.array(crop_shape) + np.array(top_left))
    crop_slice = [slice(top_left[0], bottom_right[0]),
                  slice(top_left[1], bottom_right[1])]

    # If input imagery has multiple channel extend slice so
    # all channels are in the crop.
    if multi_channel:
        crop_slice.append(slice(0, full_data.shape[-1]))

    # If input imagery is volume then crop entire stack.
    one_image_dimensions = len(crop_slice)
    if len(full_data.shape) > one_image_dimensions:
        crop_slice.insert(0, slice(0, full_data.shape[0]))

    return full_data[tuple(crop_slice)]


def normalize_channels_volume(volume):
    """
    Normalize volume input numpy array to one channel volume numpy array.
    If array is 4d then only one of them is relevant.

    Args:
        volume: numpy array with 3 or 4 dimensions
    Returns:
        3-dimension numpy array
    """
    if len(volume.shape) not in [3, 4]:
        raise ValueError("Array is neither a grayscale nor an RGB image.")

    if len(volume.shape) == 4:
        # The last dimension should be channel.
        return np.amax(volume, -1)
    return volume


def bbox(array, axes):
    """
    Find non-zero bounding boxes along the provided axes.
    Args:
        img: numpy array
        axes: axis number or list of axes

    Returns:
        first and last non-zero tuple or list such tuples representing 
        bbox or None if none found
    """
    if isinstance(axes, int):
        axes = [axes]

    res = []
    all_axis = list(range(array.ndim))
    for axis in axes:
        other_axes = tuple([a for a in all_axis if a != axis])
        non_zero_on_axis = np.any(array, axis=other_axes) if other_axes != [] else array
        where_non_zero = np.where(non_zero_on_axis)[0]
        if len(where_non_zero) == 0:
            return None
        ax_min, ax_max = where_non_zero[[0, -1]]
        res.append((ax_min, ax_max))

    if len(res) == 1:
        return res[0]
    return res


def has_gaps(array):
    """
    Determine if there are slices in bounding box of data in array that are empty.
    Args:
        array: S x Y x X numpy array

    Returns:
        True if there are empty slices
    """
    non_empty_start, non_empty_end = bbox(array, 0)
    non_empty_array = array[non_empty_start:non_empty_end + 1]
    z_existence = np.any(non_empty_array, axis=(1, 2))
    return not np.all(z_existence)


def draw_grid(im, grid_size):
    """Helper function to draw a grid on the image.
    """
    if len(im.shape) == 2:
        color = 0
    else:
        color = (0, 0, 0)

    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=color)
        for j in range(0, im.shape[0], grid_size):
            cv2.line(im, (0, j), (im.shape[1], j), color=color)


def overlay_sample_boundaries(input_volume, mask_volume, slice_idx, colour=(255, 255, 255)):
    """
    Draw contours of objects on a slice of the volume.
    Args:
        input_volume: Z x Y x X input volume which will serve as the background in overlay
        mask_volume: Z x Y x X array representing objects mask or object integer labels
        slice_idx: overlayed slice to return
        colour: colour of the contours

    Returns:
        slice from input_volume with objects from mask_volume contours overlay
    """
    input_sample = input_volume[slice_idx].copy()
    boundary_sample = skimage.segmentation.find_boundaries(mask_volume[slice_idx])
    input_sample[boundary_sample != 0] = colour
    return input_sample


def preview_augs(images, labels, generator):
    # Draw grids to preview PiecewiseAffine() transformation.
    for idx in range(len(images)):
        draw_grid(images[idx], 35)
        draw_grid(labels[idx], 35)

    if generator.mode == 'reflect':
        pad = images[0].shape[0]
    else:
        pad = 0

    img = cv2.copyMakeBorder(images[0], pad, pad, pad, pad,
                             cv2.BORDER_REFLECT)
    gt = cv2.copyMakeBorder(labels[0], pad, pad, pad, pad,
                            cv2.BORDER_REFLECT)

    seq_det = generator.seq._to_deterministic()
    im_grid = seq_det.draw_grid(img, cols=8, rows=8)
    gt_grid = seq_det.draw_grid(gt, cols=8, rows=8)
    imageio.imwrite("data/augs/im.png", im_grid)
    imageio.imwrite("data/augs/gt.png", gt_grid)


def array_2d_to_image(array, autorescale=True):
    assert array.min() >= 0
    assert len(array.shape) in [2, 3]

    if array.max() <= 1 and autorescale:
        array = 255 * array

    array = array.astype('uint8')
    array = np.squeeze(array)
    return Image.fromarray(array)


def spatial_shape(volume):
    """
    Args:
        volume: Z x Y x X or Z x Y x X x C

    Returns:
        Shape stripped from channels
            or original shape for 3D data
    """
    assert volume.ndim >= 3
    return volume.shape if volume.ndim == 3 else volume.shape[:-1]


def channels_count(volume):
    """
    Args:
        volume: Z x Y x X or Z x Y x X x C

    Returns:
        Number of channels C or None if it is 3D
    """
    assert volume.ndim >= 3
    return None if volume.ndim == 3 else volume.shape[-1]


def prepare_input_images(images, trim_method):
    image_data, _ = fix_data_dims(images, [], trim_method=trim_method,
                                  out_dim=256)

    network_input_imagery = np.expand_dims(np.array(image_data), axis=3)
    return network_input_imagery


def replace_values(array, mapping, return_copy=True, zero_unmapped=False):
    """
    Replace values using provided mapping.
    Args:
        array: numpy array
        mapping: dictionary used to replace values in array
        return_copy: should return copy or replace in place
        zero_unmapped: change the values not found in mapping to zero

    Returns:
        numpy array with some values remapped, depending on return_copy it is
        either array or new object
    """

    def complete_mapping(v):
        default = 0 if zero_unmapped else v
        return mapping.get(v, default)

    array = array.copy() if return_copy else array

    values, index = np.unique(array, return_inverse=True)
    new_values = np.array([complete_mapping(v) for v in values], dtype=array.dtype)
    array[:] = new_values[index].reshape(array.shape)
    return array


def replace_values_in_slices(slices, mapping):
    """
    Replace values using provided mapping in all provided slices.
    Assumes that once value is not present in slice (i) it will not appear
    again in slice (i+1).

    Args:
        slices: numpy arrays in order of closest to farthest
        mapping: dictionary used to replace values in slices
    Returns:
        None, always runs in place
    """

    # make sure that we do not change input dictionary
    mapping = mapping.copy()
    for s in slices:
        for k, v in list(mapping.items()):
            if k != v:
                found = s == k
                # if value k exist in current slice then replace it
                # if not then remove it from mapping as it should not occur
                # later
                if found.any():
                    s[found] = v
                else:
                    del mapping[k]


def parse_string(data):
    try:
        return int(data)
    except ValueError:
        try:
            return float(data)
        except ValueError:
            return data


def chunks(collection, chunk_size):
    for i in range(0, len(collection), chunk_size):
        yield collection[i:i + chunk_size]
