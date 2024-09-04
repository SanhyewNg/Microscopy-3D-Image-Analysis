import argparse
import os
from itertools import cycle, chain

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

from clb.yaml_utils import yaml_file, load_yaml_args
from clb.classify.extractors import extract_channels
from clb.classify.nn.train import raw_data_generator, normalizer, load_model_with_cache
from clb.dataprep.utils import rescale_to_float


def predict_cube(model, model_path, input_volumes, batch_size=None):
    """
    Use the given model (or load model from path) to classify given volumes
    Args:
        model: loaded model ready to predict
        model_path: path to h5 model, required to load model specifics from yaml
        input_volumes: collection of Z x Y x X x C or Y x X x C crops of objects matching the model
        batch_size: size of batches that go prediction, if None defaults depending on model type will be used

    Returns:
        numpy array with prediction for each object
    """
    training_params = load_yaml_args(yaml_file(model_path))
    use_3d = training_params.get("use_3d", False)

    model = model or load_model_with_cache(model_path)

    # TODO add and use here generator that can be used on both (img, gt) and (img) input stream so that
    # TODO it is consistent in prediction and training.
    input_volumes = [rescale_to_float(volume, float_type='float32') for volume in input_volumes]

    # Pick middle slice if model works on 2D slices.
    if not use_3d:
        batch_size = batch_size or 128
        if input_volumes[0].ndim == 4:
            input_volumes = [volume[len(volume) // 2] for volume in input_volumes]
    else:
        batch_size = batch_size or 32  # lower batch size so that it fits in GPU of lower end cards
        assert input_volumes[0].ndim == 4, "using 3d but input data is not 3d"

    x = np.array(input_volumes)

    predicted_classes = model.predict(x, batch_size=batch_size)
    return predicted_classes


def parse_arguments(provided_args=None):
    parser = argparse.ArgumentParser(description='CLB deep learning classification predictor.', add_help=True)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--model', help='Path to h5 model (expects also the existence of corresponding yaml files).',
                          required=True)

    input_parameters = parser.add_mutually_exclusive_group(required=True)
    input_parameters.add_argument('--input', help='Path to single crop or a directory.')
    input_parameters.add_argument('--root_input', help='Root directory with crops in datasets... subfolder.')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--filter', help='Every predicted crop should contain this text.')
    optional.add_argument('--sample_negative', dest='sample_negative', action='store_true',
                          help='Sample to predict are negative.')
    optional.add_argument('--datasets_positive', default=['1_positive'], nargs='+', type=str,
                          help='If provided specified subfolders of root should contain positive samples.')
    optional.add_argument('--datasets_negative', default=['0_negative'], nargs='+', type=str,
                          help='If provided specified subfolders of root should contain negative samples.')

    parsed_args = parser.parse_args(provided_args)

    return parsed_args


def load_data_from_folder(input_dir, filtering, expected_outcome, channels):
    """
    Load crops from folder that contain filtering string and set their gt to expected_outcome.
    Args:
        input_dir: directory with crops.
        filtering: only files that contain this string will be returned
        expected_outcome: gt class set in output list
        channels: list of channels to extract

    Returns:
        list of pairs (normalized_input_volume, gt class)
    """
    # TODO channels in crops should be present in some crop yaml definition
    one_crop = raw_data_generator(input_dir, expected_outcome, channels, filter=filtering) | normalizer()
    return list(one_crop)


def main(args):
    data_to_predict = []

    channels = extract_channels(load_yaml_args(yaml_file(args.model)).channels)
    if args.input is not None:
        # Load all data from a single directory or we just run it on a single file.
        if os.path.isfile(args.input):
            input_dir = os.path.dirname(args.input)
            filtering = os.path.basename(args.input)
        else:
            input_dir = args.input
            filtering = args.filter
        data_to_predict = load_data_from_folder(input_dir, filtering,
                                                expected_outcome=not args.sample_negative,
                                                channels=channels)
    else:
        # Load data from separate subfolders that contain only positive or negative samples.
        positive_cases = zip(args.datasets_positive, cycle([1]))
        negative_cases = zip(args.datasets_negative, cycle([0]))
        for dataset, gt_value in chain(positive_cases, negative_cases):
            data_to_predict += load_data_from_folder(os.path.join(args.root_input, dataset), args.filter,
                                                     expected_outcome=gt_value,
                                                     channels=channels)

    x, y_true = zip(*data_to_predict)

    predicted_classes = predict_cube(None, args.model, x)
    predicted_binary = predicted_classes > 0.5
    print(classification_report(y_true, predicted_binary, target_names=["negative", "positive"]))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
