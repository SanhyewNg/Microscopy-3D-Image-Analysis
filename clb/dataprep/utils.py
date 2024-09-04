import itertools as it
import operator
import os
import sys
import threading
import time
import warnings
from functools import partial
from glob import glob

import bioformats
import cv2
import imageio
import javabridge
import math
import numpy as np
import skimage
from PIL import Image

PIXELS_ATTRIBUTES = ['PhysicalSizeX', 'PhysicalSizeXUnit',
                     'PhysicalSizeY', 'PhysicalSizeYUnit',
                     'PhysicalSizeZ', 'PhysicalSizeZUnit',
                     'SizeC', 'SizeT', 'SizeX', 'SizeY', 'SizeZ', 'Type']

CHANNEL_ATTRIBUTES = ['Name', 'Color', 'ExcitationWavelength',
                      'ExcitationWavelengthUnit']


def get_tiff_paths_from_directories(data):
    """Extract paths of single TIFF files from `data`.

    Args:
        data: either a directory with multipage TIFF files or a list of
              directories to extract single TIFF file paths from.

    Returns:
        List with paths of single TIFF files.
    """
    if isinstance(data, str):
        data = [data]

    for directory in data:
        if not os.path.exists(directory):
            raise IOError('{} directory does not exist!'.format(directory))

    all_paths = [glob(os.path.join(img_dir, '*.tif')) for img_dir in data]
    single_paths = sum(all_paths, [])

    return sorted(single_paths)


def extract_label_edges(labels, boundary_thickness=2,
                        touching_boundaries=False):
    """
    Extract boundaries of cells as a mask.

    Args:
        labels: numpy array with object labels
        boundary_thickness (int): how thick is the boundary (by default it's
                                  2 pixels thick). It should always be an
                                  even number.
        touching_boundaries (bool): if True results in binary masks only with
                                    boundaries that are touching.
    Returns:
        Binary mask of edges.
    """
    if boundary_thickness % 2 != 0:
        warnings.warn('boundary_thickness should be an even number. '
                      'Passing uneven number results in different thickness.')

    # This results in boundaries that are 2 pixels thick.
    if touching_boundaries:
        boundaries = skimage.segmentation.find_boundaries(labels)
        labels_dilated = skimage.morphology.dilation(labels)
        touching_boundaries = find_touching_boundaries(labels_dilated)
        boundaries = np.logical_and(boundaries, touching_boundaries)
    else:
        boundaries = skimage.segmentation.find_boundaries(labels)

    for _ in range(2, boundary_thickness, 2):
        boundaries = skimage.morphology.binary_dilation(boundaries)

    return boundaries


def find_touching_boundaries(labels):
    """
    Find cells' edges that are touching each other.

    Args:
        labels: numpy array with object labels
    Returns:
        boundaries: Binary masks of touching edges.
    """
    boundaries = np.full(labels.shape, False, dtype=bool)

    for (x, y), current_pixel in np.ndenumerate(labels):
        if current_pixel == 0:
            continue
        # Check top.
        if x > 0:
            if labels[x - 1, y] != current_pixel and labels[x - 1, y] != 0:
                boundaries[x, y] = True
                continue

        # Check bottom.
        if x < (labels.shape[0] - 1):
            if labels[x + 1, y] != current_pixel and labels[x + 1, y] != 0:
                boundaries[x, y] = True
                continue

        # Check left.
        if y > 0:
            if labels[x, y - 1] != current_pixel and labels[x, y - 1] != 0:
                boundaries[x, y] = True
                continue
            # Check upper left
            if x > 0:
                if labels[x - 1, y - 1] != current_pixel and labels[x - 1, y - 1] != 0:
                    boundaries[x, y] = True
                    continue
            # Check lower left
            if x < (labels.shape[0] - 1):
                if labels[x + 1, y - 1] != current_pixel and labels[x + 1, y - 1] != 0:
                    boundaries[x, y] = True
                    continue

        # Check right.
        if y < (labels.shape[1] - 1):
            if labels[x, y + 1] != current_pixel and labels[x, y + 1] != 0:
                boundaries[x, y] = True
                continue
            # Check upper right
            if x > 0:
                if labels[x - 1, y + 1] != current_pixel and labels[x - 1, y + 1] != 0:
                    boundaries[x, y] = True
                    continue
            # Check lower right
            if x < (labels.shape[0] - 1):
                if labels[x + 1, y + 1] != current_pixel and labels[x + 1, y + 1] != 0:
                    boundaries[x, y] = True
                    continue

    return boundaries


def add_padding(image, pad_to_dim=256, padding_method='padding'):
    """Increase image size by adding padding.

    Args:
        image: numpy array with image to be padded
        pad_to_dim: final dimension of the image (after padding)
        padding_method: used padding method (reflect, padding)
    Returns:
        padded image
    """
    pad = int((pad_to_dim - image.shape[0]) / 2)
    BLACK = [0, 0, 0]
    if padding_method == 'padding':
        cv_method = cv2.BORDER_CONSTANT
    elif padding_method == 'reflect':
        cv_method = cv2.BORDER_REFLECT
    else:
        raise Exception("Not expected trim_method: " + padding_method)

    return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv_method,
                              value=BLACK)


def remove_padding(image, pad_to_dim):
    pad = int((pad_to_dim - image.shape[0]) / 2)
    return image[-pad: pad + image.shape[0], -pad: pad + image.shape[1]]


def fix_data_dim(image, trim_method, out_dim):
    """Resizes or adds or remove padding to image to make image dimensions
    be 2^n.
        Args:
            image: square image to pad
            trim_method: method to fix dims (reflect, padding, resize)
            out_dim: desired dimension of padded image

        Returns:
            A padded / unpadded image

        Raises:
            ValueError when the array has values larger than 1.0 and is of
            floating point type.
    """
    if trim_method == 'resize':
        output = cv2.resize(image, (out_dim, out_dim))
    else:
        if out_dim >= image.shape[0]:
            output = add_padding(image, out_dim, trim_method)
        else:
            output = remove_padding(image, out_dim)
    return output


def fix_data_dims(images, masks, trim_method, out_dim=256):
    """Resizes or adds or remove padding to images to make image dimensions
    be 2^n.
    Args:
        images: either a list of images or a numpy array with images to pad
        masks: either a list of images or a numpy array with masks to pad
        trim_method: method to fix dims (reflect, padding, resize)
        out_dim: desired dimension of padded images and masks

    Returns:
        A tuple of padded / unpadded images and masks
    """
    return (
        np.array([fix_data_dim(img, trim_method, out_dim) for img in images]),
        np.array([fix_data_dim(mask, trim_method, out_dim) for mask in masks]))


def rescale_to_float(array, float_type='float64'):
    """
    Given array with values bigger than 1 rescale it to 0-1.
    Args:
        array: numpy array 2d
        float_type: float type to use in conversion
    Returns:
        numpy array with values in 0-1
    """
    if np.amax(array) > 1:
        return array.astype(float_type) / np.iinfo(array.dtype).max
    return array.astype(float_type)


def get_number_of_pages(file):
    """Get number of pages in TIFF file.

    Args:
        file: path to the TIFF file (can be either a single-page or a
              multi-page).

    Returns:
        Number of pages from the TIFF file.
    """
    with Image.open(file) as img:
        if hasattr(img, 'n_frames'):
            return img.n_frames
        else:
            return 1


def load_tiff_stack(path, channel_merger=None):
    """Loads TIFF file (single or multi page).

    There are four cases that this function is able to handle:
    1) single-page grayscale TIFF (height, width)
    2) single-page multi-channel TIFF (height, width, channels)
    3) multi-page grayscale TIFF (pages, height, width)
    4) multi-page multi-channel TIFF (pages, height, width, channels)

    Args:
        path: path to TIFF file
        channel_merger: method to merge multi channel image.
            Default is to pick the channel with most intensity
            (works with LUT imagery).

    Returns:
        Output volume of shape (num_pages, height, width).
    """
    channel_merger = channel_merger or reduce_to_max_channel

    img = imageio.volread(path)
    n_pages = get_number_of_pages(path)

    if len(img.shape) == 2:
        # case 1)
        return np.array([img])

    elif len(img.shape) == 3 and n_pages == 1:
        # case 2)
        return np.array([channel_merger(img)])

    elif len(img.shape) == 3 and n_pages > 1:
        # case 3)
        return img

    elif len(img.shape) == 4:
        # case 4)
        return np.array([channel_merger(page) for page in img])


def reduce_to_max_channel(array):
    """
    Normalize input numpy array to one channel numpy array.
    If array is 3d then only one of them is relevant.
    Args:
        array: numpy array with 2 or 3 dimensions
    Returns:
        2-dimension numpy array
    """
    if len(array.shape) not in [2, 3]:
        raise ValueError("Array is neither a grayscale nor an RGB image.")

    if len(array.shape) == 3:
        # The smallest dimension should be channels.
        channel_axis, _ = min(enumerate(array.shape),
                              key=operator.itemgetter(1))
        return np.amax(array, channel_axis)
    return array


def pick_channel(channel, array):
    """Convert input numpy array to one channel numpy array by
    picking one of the channels.
    Args:
        array: numpy array with 2 or 3 dimensions
        channel: channel to return
    Returns:
        2-dimension numpy array
    """
    if len(array.shape) not in [2, 3]:
        raise ValueError("Array is neither a grayscale nor an RGB image.")

    if len(array.shape) == 3:
        return array[::, ::, channel]

    return array


class FileTypeError(Exception):
    """Raised when file extension is not .lif nor .tif"""


def bioformats_opener(path):
    """Return bioformats reader.

    Function starts java virtual machine and schedules its closing. Then it
    opens bioformats.ImageReader.

    Args:
        path (str): Path to file.
    """
    _start_vm()
    reader = bioformats.ImageReader(path)

    return reader


def kill_vm_excepthook(exc_type, exc_value, traceback):
    """Function kills java vm and runs original excepthook.

    Args:
        exc_type: Type of an exception.
        exc_value: Value of an exception.
        traceback:
    """
    javabridge.kill_vm()
    sys.__excepthook__(exc_type, exc_value, traceback)


def _start_vm(watcher_name='watcher', watcher_interval=2, log_level='WARN'):
    """Start virtual machine and thread watching it, if it's not started.

    Thread is checking every `watcher_interval` if main thread is still running
    and if it's not it kills virtual machine. See _watch function. Function
    also sets logging level. Method of setting logging level is from this
    discussion:
    forum.image.sc/t/python-bioformats-and-javabridge-debug-messages/12578/12

    Args:
        watcher_name (str): Name of the watching thread.
        watcher_interval (int): Time (in seconds) between two checks.
        log_level (str): Logging level
    """
    if not _is_thread_running(watcher_name):
        javabridge.start_vm(class_path=bioformats.JARS)
        # Making sure, that vm is killed when exception is not caught.
        sys.excepthook = kill_vm_excepthook
        logger_name = javabridge.get_static_field("org/slf4j/Logger",
                                                  "ROOT_LOGGER_NAME",
                                                  "Ljava/lang/String;")
        logger = javabridge.static_call("org/slf4j/LoggerFactory",
                                        "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                        logger_name)
        log_level = javabridge.get_static_field("ch/qos/logback/classic/Level",
                                                log_level,
                                                "Lch/qos/logback/classic/Level;")
        javabridge.call(logger, "setLevel",
                        "(Lch/qos/logback/classic/Level;)V", log_level)

        watch = partial(_watch, check_interval=watcher_interval)
        watcher = threading.Thread(target=watch, name='watcher')
        watcher.start()


def _watch(check_interval=2):
    """Watch main thread and kill vm when it terminates.

    Args:
        check_interval (int): Time (in seconds) between two checks if main
                              thread is alive.
    """
    while threading.main_thread().is_alive():
        time.sleep(check_interval)

    javabridge.kill_vm()


def _is_thread_running(name):
    """Check if thread with given `name` is alive.

    Function just checks if thread `name` is in threads in threading.enumerate.

    Args:
        name (str): Thread name.

    Returns:
        bool: True if thread is running, False otherwise.
    """
    threads = threading.enumerate()
    threads_names = [thread.name for thread in threads]

    return name in threads_names


def read_volume(path, channels=None, series=0):
    """Read volume of images from .tif or .lif file.

    When it's reading from one channel output shape should be same as in
    `load_tiff_stack`.

    Args:
        path (str): Path to file.
        channels (int or list): Channel(s) to read. If None read all. It only
                                applies to .lif files.
        series (int): Series to read from. It only applies to .lif files.

    Returns:
        np.ndarray: Read volume.
    """
    ext = os.path.splitext(path)[1]
    if ext in {'.tif', '.tiff'}:
        if (channels is not None) or series:
            raise Exception('Used arguments for .lif file, but file is .tif.')
        volume = load_tiff_stack(path)
    elif ext == '.lif':
        volume = imageio.volread(path, channels=channels, series=series)
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
    else:
        raise Exception('Wrong file format (neither .tif or .lif).')

    return volume


class SeriesError(Exception):
    """Raised when trying to read non existing series."""


class PixelSizeError(Exception):
    """Raised when x pixel size is different than y pixel size."""


def calc_desired_shape(shape, current_x, current_y, desired_x, desired_y):
    """Calculate shape to which image has to be rescaled, so that pixel size is
       equal to `desired`.

    Function assumes that both x and y pixel size are the same, so it's caller
    responsibility to check that.

    Args:
        shape (tuple): Current shape of the image.
        current_x (float): Current pixel size in X dimension.
        current_y (float): Current pixel size in Y dimension.
        desired_x (float): Desired pixel size in X dimension.
        desired_y (float): Desired pixel size in X dimension.

    Returns:
        tuple: New shape of the image.
    """
    new_img_size = np.array(shape[:2])
    new_img_size[1] = current_x * new_img_size[1] // desired_x
    new_img_size[0] = current_y * new_img_size[0] // desired_y
    new_img_size = tuple(int(x) for x in new_img_size)

    return new_img_size


def save_to_tif(path, data):
    """Save `data` to tif file.

    For now this function always saves images as uint8, because there were some
    problems with saving uint16.

    Args:
        path (str): Path to tif file.
        data (np.ndarray): Image to save. Should be of shape (Z, Y, X, C).
    """
    _start_vm()

    for z, image in enumerate(data):
        for c in range(image.shape[-1]):
            bioformats.write_image(path,
                                   image[..., c],
                                   'uint8',
                                   c=c,
                                   z=z,
                                   size_c=image.shape[-1],
                                   size_z=data.shape[0])


def istack(arrays, start=0, stop=None):
    """Stack iterable of arrays in one array.

    Args:
        arrays (Iterable): Arrays to be stacked.
        start (int): Starting element.
        stop (int/None): Element after the last element.

    Returns:
        np.ndarray: Stacked arrays.
    """
    sliced = it.islice(arrays, start, stop)
    stacked = np.stack(sliced)

    return stacked


def resize_to_pixel_size(image, pixel_size, desired_pixel_size):
    """Resize `image`, to make it match `desired_pixel_size`.

    Args:
        image (np.ndarray): Image, shape (Y, X).
        pixel_size (tuple): Current pixel size of `image`. First element is
        X dim, second is Y dim.
        desired_pixel_size (tuple): Desired pixel size.First element is X dim,
        second is Y dim.

    Returns:
        np.ndarray: Resized image, same shape as `image`.
    """
    new_img_shape = calc_desired_shape(image.shape,
                                       current_x=pixel_size[0],
                                       current_y=pixel_size[1],
                                       desired_x=desired_pixel_size[0],
                                       desired_y=desired_pixel_size[1])
    image = cv2.resize(image, new_img_shape)

    return image


def ensure_pixel_size(volume, desired_pixel_size, tolerance, pixel_size=None):
    """Ensure that `volume` has right pixel size.

    If `volume` has pixel size that differs from `desired_pixel_size` by more
    than `tolerance` `volume` is resized.

    Args:
        volume (VolumeIter): Image slices, shape (Y, X).
        desired_pixel_size (tuple): Desired pixel size of the image.
            It's a two element tuple, containing physical X and Y length
            of the pixel. X is first.
        tolerance (float): Pixel size tolerance.
        pixel_size (tuple): Pixel size of the image, defined by two floats.
            First is length in X dimension, then in Y dim.
    Returns:
        tuple: volume (VolumeIter), was it resized (bool).
    """
    resized = False

    # Metadata has a priority over CLI.

    try:
        size_x = float(volume.metadata['PhysicalSizeX'])
        size_y = float(volume.metadata['PhysicalSizeY'])
        pixel_size = (size_x, size_y)
    except KeyError:
        pass

    if pixel_size is None:
        raise Exception("Pixel size is neither specified by input or metadata,"
                        "no resize has been done.")
    else:
        assert len(pixel_size) == 2, \
            "Pixel size has to be two-element, iterable object"
        assert len(desired_pixel_size) == 2 and \
               desired_pixel_size is not None, \
            "Desired pixel size has to be two-element, iterable object"

        if not (math.isclose(pixel_size[0], desired_pixel_size[0],
                             rel_tol=0, abs_tol=tolerance) and
                math.isclose(pixel_size[1], desired_pixel_size[1],
                             rel_tol=0, abs_tol=tolerance)):
            volume = volume.transform(resize_to_pixel_size, pixel_size,
                                      desired_pixel_size)
            resized = True

    return volume, resized


class DimensionError(Exception):
    """Raised when input image has wrong dimensionality."""


class FileFormatError(Exception):
    """Raised when output file has wrong extension."""


class UFFLocationError(Exception):
    """Raised when location for .uff file is wrong."""


def ensure_path(path, extensions, assume_uff=True):
    """Create dirname of `path`.

    Function checks if dirname of `path` exists and if basename of `path` has
    right extension. If basename has extension that is not in `extensions` file
    and `assume_uff` is True, file is assumed to be .uff.
    Then given path should point to existing empty directory or non existing
    file. If `assume_uff` is False file extension should be in `extensions`.

    Args:
        path (str): Path to validate.
        extensions (tuple|str): Acceptable extensions.
        assume_uff (bool):

    Raises:
        UFFLocationError: If `path` is assumed to be .uff file and it points to
                          file or non empty directory.
        FileFormatError: If `assume_uff` is False and file extension is not
                         in `extensions`.
    """
    dirname, basename = os.path.split(os.path.abspath(path))
    os.makedirs(dirname, exist_ok=True)

    # Output is assumed to be .uff, so `basename` directory should be empty
    # or not exist.
    if not basename.endswith(extensions):
        if assume_uff:
            try:
                if os.listdir(path):
                    raise UFFLocationError('Directory {} is not empty.'
                                           .format(path))
            except NotADirectoryError:
                raise UFFLocationError('File {} exists and is not a directory.'
                                       .format(path))
            except FileNotFoundError:
                pass
        else:
            raise FileFormatError('Wrong file extension, expected one of {}'
                                  .format(extensions))


def parse_omexml_metadata(image):
    """Parse bioformats.OMEXML.Image object into dictionary

    Args:
        image (bioformats.OMEXML.Image): Metadata of series.

    Returns:
        dict: Parsed metadata.
    """
    metadata = {'Name': image.Name}

    pixels = image.Pixels
    pixels_attrib = pixels.node.attrib
    for attr in PIXELS_ATTRIBUTES:
        if attr in pixels_attrib:
            metadata[attr] = pixels_attrib[attr]

    n_channels = image.Pixels.get_SizeC()
    channels_attrib = [pixels.Channel(i).node.attrib for i in
                       range(n_channels)]
    metadata['Channels'] = [dict() for _ in channels_attrib]
    for i, channel_attrib in enumerate(channels_attrib):
        for attr in CHANNEL_ATTRIBUTES:
            if attr in channel_attrib:
                metadata['Channels'][i][attr] = channel_attrib[attr]
    return metadata


def ensure_2d_rgb(image):
    """Change the shape to (Y, X, 3) from (Y, X) or (Y, X, 2)."""
    assert 3 >= len(image.shape) >= 2, 'shape len < 2 or > 3'

    if len(image.shape) == 2:
        image = np.expand_dims(image, 2)

    channel_count = image.shape[-1]
    assert channel_count <= 3, 'more than 3 channels'
    if channel_count == 3:
        return image
    else:
        return np.pad(image, ((0, 0), (0, 0), (0, 3 - channel_count)), 'constant', constant_values=0)


def ensure_3d_rgb(image):
    """Change the shape to (N, Y, X, 3) format.

    Args:
        image: numpy of shape (N, Y, X), (N, Y, X, 1), (N, Y, X, 2) or (N, Y, X, 3)

    Returns:
        np.ndarray: Reshaped image.
    """
    assert 4 >= len(image.shape) >= 3, 'shape len < 3 or > 4'

    if len(image.shape) == 3:
        image = np.expand_dims(image, 3)

    channel_count = image.shape[-1]
    assert channel_count <= 3, 'more than 3 channels'
    if channel_count == 3:
        return image
    else:
        return np.pad(image, ((0, 0), (0, 0), (0, 0), (0, 3 - channel_count)), 'constant', constant_values=0)


def ensure_shape_4d(shape, assume_missing_channel=True):
    """Change the shape to (N, Y, X, C) format.

    Args:
        shape (array_like): It should be 1d.
        assume_missing_channel: Should function assume that C is missing in case of
                                length 3 shape.

    Returns:
        tuple: New shape.
    """
    assert 4 >= len(shape) >= 2, 'shape len < 2 or > 4'

    # Trimming ones.
    shape = tuple(np.trim_zeros(np.array(shape) - 1) + 1)

    if len(shape) == 2:
        shape = (1, *shape, 1)
    elif len(shape) == 3:
        if assume_missing_channel:
            shape = (*shape, 1)
        else:
            shape = (1, *shape)

    return shape


def ensure_4d(image, assume_missing_channel=True):
    """Reshape image to (N, Y, X, C).

    It is assumed that Y and X dimensions are bigger than 1.

    Args:
        image (array_like): Image, possibly with N and C dimension missing.
        assume_missing_channel (bool): Should function assume that channel is
                                       missing in case of 3d image. If it's
                                       True dimension is added at the end,
                                       otherwise at the beginning.

    Returns:
        np.ndarray: Reshaped image.
    """
    new_shape = ensure_shape_4d(image.shape, assume_missing_channel)

    return image.reshape(new_shape)
