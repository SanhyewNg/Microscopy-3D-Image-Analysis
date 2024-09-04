import os
from functools import lru_cache

import numpy as np
from keras.models import load_model

import clb.train.metrics as metrics
import clb.train.losses as losses
from clb.dataprep.utils import fix_data_dims, rescale_to_float
from clb.image_processing import separate_objects
from clb.networks.dcan import (get_probability_from_sigmoid,
                                get_probability_from_tanh)
from clb.utils import prepare_input_images
from clb.yaml_utils import load_yaml_args


class HDict(dict):
    """Hashable dictionary.
    """
    def __hash__(self):
        return hash(frozenset(self.items()))


@lru_cache(maxsize=10)
def load_model_with_cache(model_path):
    try:
        return load_model(model_path)
    except ValueError:
        # Model probably contains custom metric, load it with this metric
        # instead.
        custom_objects = HDict({'iou': metrics.iou,
                                    'weighted_crossentropy':
                                        losses.weighted_crossentropy,
                                    'background_recall':
                                        metrics.channel_recall(channel=0,
                                                               name=
                                                               "background_recall"),
                                    'background_precision':
                                        metrics.channel_precision(channel=0,
                                                                  name=
                                                                  "background_precision"),
                                    'objects_recall':
                                        metrics.channel_recall(channel=1,
                                                               name=
                                                               "objects_recall"),
                                    'objects_precision':
                                        metrics.channel_precision(channel=1,
                                                                  name=
                                                                  "objects_precision"),
                                    'boundaries_recall':
                                        metrics.channel_recall(channel=2,
                                                               name=
                                                               "boundaries_recall"),
                                    'boundaries_precision':
                                        metrics.channel_precision(channel=2,
                                                                  name=
                                                                  "boundaries_precision"),
                                    'background_iou':
                                        metrics.channel_iou(channel=0,
                                                            name=
                                                            "background_iou"),
                                    'objects_iou':
                                        metrics.channel_iou(channel=1,
                                                            name="objects_iou"),
                                    'boundaries_iou':
                                        metrics.channel_iou(channel=2,
                                                            name=
                                                            "boundaries_iou")})
        return load_model(model_path, custom_objects=custom_objects)

@lru_cache(maxsize=10)
def get_probability_calculator(model_path):
    yaml_path = os.path.splitext(model_path)[0] + ".yaml"

    dcan_final_act = None
    if os.path.isfile(yaml_path):
        args = load_yaml_args(yaml_path)
        dcan_final_act = args.get('dcan_final_act')

    if dcan_final_act is None:
        return get_probability_from_sigmoid
    elif dcan_final_act == 'tanh':
        return get_probability_from_tanh
    elif dcan_final_act == 'sigmoid':
        return get_probability_from_sigmoid
    else:
        raise KeyError("invalid dcan_final_act: {0}", dcan_final_act)


def predict_dcan(images, model_path, trim_method, postprocess=False):
    """Function making use of DCAN architecture to predict images.

    Network is trained on the data with following assumptions:

        - x is a numpy array of type `float64` normalized with
        rescale_to_float. [num_images, dim_x, dim_y, 1] (4D tensor required
        by Keras pipeline),
        - y is a numpy array of type `float64` normalized with
        rescale_to_float. [num_images, dim_x, dim_y, 2] (channel 0 are
        objects, where '0' is background and '1' is objects and channel 1 is
        for boundaries, where '0' is background and '1' is boundary).

    It was created on @Fafa87's request.

    Args:
        images: input images to perform prediction on
        model_path: path to trained model
        trim_method: method used to fit into 256 (padding, reflect, resize)
        postprocess: should postprocess probabilities and calculate instance
    Returns:
        predictions or instance segmentation
    """
    model = load_model_with_cache(model_path)

    rescaled_images = list(map(rescale_to_float, images))
    network_ready_images = prepare_input_images(rescaled_images, trim_method)

    objects, boundaries = map(np.squeeze,
                              model.predict(network_ready_images,
                                            batch_size=8,
                                            verbose=0))

    prob_calculator = get_probability_calculator(model_path)
    pred_objects = prob_calculator(objects)
    pred_boundaries = prob_calculator(boundaries)

    if postprocess:
        # For DCAN it's necessary to run separate_objects to include boundaries
        # information in instance-aware segmentation.
        masks = [separate_objects(obj, bnd) for obj, bnd in zip(pred_objects, pred_boundaries)]
    else:
        # Probability of being a cell and not being a boundary.
        masks = pred_objects * (1 - pred_boundaries)

    # We assume that images are squares.
    return fix_data_dims(masks, [], trim_method=trim_method, out_dim=rescaled_images[0].shape[0])[0]
