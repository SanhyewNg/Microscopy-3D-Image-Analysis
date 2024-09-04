from collections import OrderedDict

import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude

from clb.dataprep.utils import rescale_to_float
from clb.image_processing import clahe, extend_membrane, estimate_membrane_from_nucleus

DEFAULT_PREPROCESSING_PARAMS = {
    "clahe": {'size': 70, 'median_size': 2, 'clip_limit': 0.015},
    "edges": {'gaussian_sigma': 1.5},
    "memb": {'scale': 1}
}

DESIRED_VOXEL_SIZE = (0.5, 0.5, 0.5)
DESIRED_VOXEL_UM = np.prod(DESIRED_VOXEL_SIZE)


def parse_channels_preprocessing(channels_preprocessing_string):
    """
    Parse string into list of preprocessings.
    Args:
        channels_preprocessing_string: string with list of channel preprocessings in format:
            "1,3,4-clahe"

    Returns:
        list of preprocessings
    """
    return channels_preprocessing_string.split(',')


def extract_channels(channels_preprocessing_list):
    """
    Extract unique channel numbers from channels with preprocessings list
    Args:
        channels_preprocessing_list: string with ',' or list of strings with channels (potentially with preprocessing suffix)

    Returns:
        list of unique ints representing channels existing in channels_preprocessing_list
    """
    if isinstance(channels_preprocessing_list, str):
        channels_preprocessing_list = parse_channels_preprocessing(channels_preprocessing_list)
    return list(OrderedDict.fromkeys([int(cp.split("-")[0]) for cp in channels_preprocessing_list]))


def preprocess_channel(channel_volume, labels_volume, preprocessing, params=None, voxel_size=DESIRED_VOXEL_SIZE):
    """
    Preprocess channel_volume according to state preprocessing.
    Args:
        channel_volume: S x Y x X volume with values normalized to 0-1
        labels_volume: S x Y x X volume of labels
        preprocessing: name of the preprocessing to use,
                        one of: clahe, edges, memb
        params: complete set of parameters used in preprocessings
                see DEFAULT_PREPROCESSING_PARAMS for details
        voxel_size: tuple with the real world size of the voxel (z,y,x), especially important to normalize Z axis which usually
            has much lower resolution (voxel is large in Z axis)

    Returns:
        pair of S x Y x X volumes preprocessed accordingly:
            - preprocessed channel_volume
            - preprocessed labels_volume
    """
    assert (channel_volume.dtype == np.float32 or channel_volume.dtype == np.float64) and np.min(
        channel_volume) >= 0 and np.max(channel_volume) <= 1
    params = params or DEFAULT_PREPROCESSING_PARAMS

    preproc_params = params.get(preprocessing, None)
    # TODO we need to think whether these parameters should be scaled by pixel-size or not
    if preprocessing == 'clahe':
        clahe_slices = [clahe(s, preproc_params['size'], preproc_params['median_size'],
                              clip_limit=preproc_params['clip_limit']) for s in channel_volume]
        return np.array(clahe_slices), labels_volume
    elif preprocessing == 'edges':
        mag_slices = [gaussian_gradient_magnitude(s, preproc_params['gaussian_sigma']) for s in channel_volume]
        return np.array(mag_slices), labels_volume
    elif preprocessing == 'memb':
        labels_on_edges = [estimate_membrane_from_nucleus(s, preproc_params['scale']) for s in labels_volume]
        extended_slices = [extend_membrane(s, preproc_params['scale']) for s in channel_volume]
        return np.array(extended_slices), np.array(labels_on_edges)
    raise KeyError("{0} is not supported preprocessing.".format(preprocessing))


def preprocess_input_labels(images_volume, labels_volume, channel_with_preprocess, voxel_size=DESIRED_VOXEL_SIZE):
    """
        Preprocess input imagery and cells labels accordingly to the specified preprocessing.
        It also ensures that channel_volume is float32.
        Args:
            images_volume: S x Y x X x C or S x Y x X
                with channel for which we want to calculate features
            labels_volume: S x Y x X
                cell level segmentation as cell labels
            channel_with_preprocess: channels with optional preprocessings to use e.g. "1" or "1-equal"
            voxel_size: tuple with the real world size of the voxel (z,y,x),
                especially important to normalize Z axis which usually
                has much lower resolution (voxel is large in Z axis)
                if None then no voxel resizing should be done
        Returns:
            tuple of preprocessed channel rescaled to float32
                    and
                    preprocessed cell labels volume
    """
    channel_preprocess = channel_with_preprocess.split("-")
    channel = int(channel_preprocess[0])

    if images_volume.ndim > 3:
        preprocessed_channel_volume = images_volume[..., channel]
    else:
        preprocessed_channel_volume = images_volume[:]

    preprocess_labels_volume = labels_volume
    preprocessed_channel_volume = rescale_to_float(preprocessed_channel_volume, float_type='float32')
    if len(channel_preprocess) > 1:
        preprocessed_channel_volume, preprocess_labels_volume = preprocess_channel(preprocessed_channel_volume,
                                                                                   preprocess_labels_volume,
                                                                                   channel_preprocess[1],
                                                                                   voxel_size=voxel_size)

    return preprocessed_channel_volume, preprocess_labels_volume
