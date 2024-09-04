import argparse
import os
import random
from functools import lru_cache

import imageio
import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import load_model

import clb.train.tbcallbacks as tbc
import clb.classify.extractors as ex
from clb.dataprep.utils import get_tiff_paths_from_directories, rescale_to_float
from clb.networks import vgg
from clb.predict.predict_tile import HDict
from clb.train.metrics import f1
from clb.yaml_utils import save_args_to_yaml
from clb.dataprep.augmenter import augmentations_generator
from vendor.genny.genny.wrappers import gen_wrapper


@lru_cache(maxsize=10)
def load_model_with_cache(model_path):
    try:
        return load_model(model_path)
    except ValueError:
        # Model probably contains custom metric, load it with this metric instead.
        custom_objects = HDict({'f1': f1})
        return load_model(model_path, custom_objects=custom_objects)


@gen_wrapper
def raw_data_generator(image_data,
                       gt_class,
                       channels,
                       shuffle=False,
                       filter=None,
                       infinite=False,
                       number=None,
                       use3d=False):
    """Generate raw data from directories.

    Args:
        image_data (path): directory with multipage TIFF files with images
                    or a list of directories to generate images from
        gt_class (int): class of all read samples
        channels (list): list of channels to extract from imagery
        shuffle (bool): shuffle all the data
        filter (string): text which each file has to contain
        infinite (bool): if True it iwll loop infinitely
        number (int): if not None then it limits the number of files processed
        use3d (bool): if False only the middle slice of the imagery will be returned

    Yields:
        tuple (image, ground truth)
    """

    img_files = [p for p in get_tiff_paths_from_directories(image_data) if
                 filter is None or filter in os.path.basename(p)]

    if shuffle:
        np.random.shuffle(img_files)

    output_size = 0
    while True:
        for img_file in img_files:
            # Original implementation is much faster when there is a loooot of files.
            img_stack = imageio.volread(img_file, format="TIFF")
            img_stack = img_stack[..., channels]

            classes = gt_class
            if not use3d:
                img_stack = img_stack[len(img_stack) // 2]

            yield (img_stack, classes)
            output_size += 1

            if number is not None and number <= output_size:
                return

        if not infinite:
            break


"""
TODO we could add decorator of such sort that will allow to use same generator for both training and prediction
def process_if_predicted(data_gen, processing):
    for img_gt in data_gen:
        if not isinstance(img_gt, tuple) or not isinstance(img_gt, list):
            img_gt = [img_gt]
        img_gt = list(img_gt)
        img_gt[0] = processing(img_gt)
        yield img_gt
"""


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
        yield (norm_img, gt)


def parse_arguments(provided_args=None):
    parser = argparse.ArgumentParser(description='CLB deep learning classification training.', add_help=True)

    required = parser.add_argument_group('required arguments')
    required.add_argument('--class_name', help='name of the class to predict used to select annotations', required=True)
    required.add_argument('--model', help='path (with name) to save the model')
    required.add_argument('--channels', help='list of channels to use from the data (e.g. "0,2")', required=True)

    required.add_argument('--training_negative', help='path to negative training crops')
    required.add_argument('--training_positive', help='path to positive training crops')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--seed', type=int, help='set seed for all random generators', default=42)
    optional.add_argument('--tb_dir', default='tensorboard_vgg/',
                          help='output for tensorboard logs, {model} will be replaced by model name')
    optional.add_argument('--load_weights', help='path to model from which initialize weights.')

    dataset = parser.add_argument_group('dataset arguments')
    dataset.add_argument('--crops_substring', help='substring in crop files')
    dataset.add_argument('--use_cubes', help='size of the cropped cubes (in um)', default=16)
    dataset.add_argument('--size_to_load', type=int, default=None, help='how much crops to load from the location')
    dataset.add_argument('--use_3d', dest='use_3d', action='store_true')

    hyperparams = parser.add_argument_group('hyper parameters')
    hyperparams.add_argument('--optimizer', default='optimizers.Adam(lr=args.lr)')
    hyperparams.add_argument('--epochs', type=int, default=100)
    hyperparams.add_argument('--lr', type=float, default=1e-3)
    hyperparams.add_argument('--batch_size', type=int, default=100)

    architecture = parser.add_argument_group('architecture parameters')
    architecture.add_argument('--vgg_filters', type=int, default=32, help='number of filters in the first layer')
    architecture.add_argument('--vgg_batchnorm', type=float, default=None,
                              help='batch norm momentum value, no normalization is not specified')
    architecture.add_argument('--vgg_dropout', type=float, default=None,
                              help='dropout value used, no dropout if not specified')

    parsed_args = parser.parse_args(provided_args)
    parsed_args.no_voxel_resize = False  # be consistent with feature based training

    parsed_args.method = "DL"
    return parsed_args


def main(args):
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    tb_dir = args.tb_dir
    if args.model:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        tb_dir = tb_dir.format(model=model_name)

    os.makedirs(tb_dir, exist_ok=True)

    channels = ex.extract_channels(args.channels)
    negative = list(raw_data_generator(args.training_negative, 0, channels,
                                       shuffle=True, filter=args.crops_substring,
                                       number=args.size_to_load, use3d=args.use_3d) | normalizer())
    positive = list(raw_data_generator(args.training_positive, 1, channels,
                                       filter=args.crops_substring,
                                       number=args.size_to_load,
                                       use3d=args.use_3d) |
                    augmentations_generator() | normalizer())

    positive_weight = len(negative) / len(positive)
    print("Positive: {0}, negative {1}, positive_weight: {2}".format(len(positive), len(negative), positive_weight))

    # TODO take visualisation from validation set!
    # TODO make validation set manually before training
    small_sample_visualize = random.sample(negative, 40) + random.sample(positive, 40)
    small_sample_visualize_x, small_sample_visualize_y = zip(*small_sample_visualize)
    small_sample_visualize_x = np.array(small_sample_visualize_x)

    sample_visualize = random.sample(negative, 150) + random.sample(positive, 150)
    sample_visualize_x, sample_visualize_y = zip(*sample_visualize)
    sample_visualize_x = np.array(sample_visualize_x)

    # Callbacks.
    cbs = []
    cbs.append(ReduceLROnPlateau(monitor='val_binary_crossentropy', factor=1e-1, mode='min',
                                 patience=14, min_lr=1e-6, verbose=1))
    cbs.append(EarlyStopping(monitor='val_binary_crossentropy', min_delta=1e-4, mode='min',
                             patience=40, verbose=1))
    cbs.append(tbc.TensorBoard(log_dir=tb_dir))
    cbs.append(CSVLogger(filename=os.path.join(tb_dir, "log.csv")))

    if args.model:
        model_path_prefix = os.path.splitext(args.model)[0] + "_class_" + args.class_name
        save_args_to_yaml(model_path_prefix, args)

        # TODO save one of the models where model argument points
        cbs.append(ModelCheckpoint(filepath=model_path_prefix + '-best-val-loss.h5',
                                   monitor='val_loss',
                                   mode='min',
                                   save_best_only=True,
                                   verbose=1))
        cbs.append(ModelCheckpoint(filepath=model_path_prefix + '-best-val-f1.h5',
                                   monitor='val_f1',
                                   mode='max',
                                   save_best_only=True,
                                   verbose=1))
        cbs.append(tbc.Classification(small_sample_visualize_x, small_sample_visualize_y,
                                      name='predictions', tiles_height=5,
                                      log_dir=os.path.join(tb_dir, 'predictions')))
        cbs.append(tbc.Classification(sample_visualize_x, sample_visualize_y, only_errors=True,
                                      name='prediction_errors', tiles_height=10,
                                      log_dir=os.path.join(tb_dir, 'prediction_errors')))

    optimizer = eval(args.optimizer)
    model = vgg.VGG(negative[0][0].shape,
                    filters=args.vgg_filters,
                    batch_norm=args.vgg_batchnorm,
                    dropout=args.vgg_dropout).build()

    # TODO add precision and recall metrics
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'binary_accuracy', f1])
    model.summary()

    both_shuffled = negative + positive
    np.random.shuffle(both_shuffled)

    x = np.array(list(zip(*both_shuffled))[0])
    y = np.array(list(zip(*both_shuffled))[1])

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    model.fit(x=x, y=y, class_weight=[positive_weight],
              batch_size=args.batch_size, epochs=args.epochs, verbose=2, validation_split=0.3, callbacks=cbs)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
