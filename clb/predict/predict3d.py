import argparse
import functools as ft
import os

import cv2
import imageio
import keras.models as models
from tqdm import tqdm

import clb.dataprep.readers as readers
import clb.denoising.denoise as denoise
from clb.dataprep.utils import ensure_pixel_size
from clb.predict.predict2d import (segmentation2d_by_tiles,
                                   segmentation2d_dcan_tile)
from clb.predict.predict_tile import load_model_with_cache
from clb.yaml_utils import load_yaml_args


def predict(input_slices, model, model_args):
    """Use `model` for prediction on `input_slices`.

    Args:
        input_slices (Iterable): Model input.
        model (str): Path to .h5 file with model.
        model_args (AttrDict): Model parameters.

    Returns:
        Map object with predicted slices.
    """
    segmentation_tile = ft.partial(segmentation2d_dcan_tile,
                                   trim_method=model_args.trim_method,
                                   postprocess=False,
                                   model_path=model)
    segment2d = ft.partial(segmentation2d_by_tiles,
                           pad_size=30,
                           segmentation_tile=segmentation_tile,
                           tile_size=140)

    output_slices = map(segment2d, input_slices)

    return output_slices


def segmentation3d_by_2d(input_volume, model, model_args, tolerance,
                         pixel_size=(0.5, 0.5), desired_pixel_size=(0.5, 0.5),
                         no_pixel_resize=False):

    resized = no_pixel_resize
    args = load_yaml_args(model_args)

    if not resized:
        input_volume, resized = \
            ensure_pixel_size(input_volume,
                              pixel_size=pixel_size,
                              desired_pixel_size=desired_pixel_size,
                              tolerance=tolerance)

    preprocessed_slices = preprocess(input_volume,
                                     args.get('preprocessings', []))
    output_slices = predict(preprocessed_slices, model, args)

    if resized:
        output_slices = (cv2.resize(s, input_volume.shape[1:3]) for s in
                         output_slices)

    return output_slices


def parse_arguments():
    return get_parser().parse_args()


def get_parser():
    parser = argparse.ArgumentParser(description='CLB segmentation 3D.',
                                     add_help=False)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--input', help='3d input tiff path', required=True)
    required.add_argument('--output', help='output path', required=True)
    required.add_argument('--trim_method',
                          help=('method of adapting input size for network '
                                '(resize, padding, reflect)'),
                          default="padding")
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--model', help='path to network model')
    optional.add_argument('--model_args',
                          help='path to yaml file with training parameters')
    optional.add_argument('--series',
                          help='Series of images to read (applies'
                               ' only to .lif files).',
                          type=int,
                          default=0)
    optional.add_argument('--channel',
                          help='Index of channel to read from, '
                               'if not specified channels will be merged.',
                          type=int, default=None)
    optional.add_argument('--start',
                          type=int,
                          default=0,
                          help='Starting slice for segmentation.')
    optional.add_argument('--stop',
                          type=int,
                          default=None,
                          help='Last slice for segmentation. It is not'
                               'segmented. If not given, all slices from the'
                               'starting slice are segmented.')
    optional.add_argument('--pixel_size',
                          type=float,
                          nargs='+',
                          default=(0.5, 0.5),
                          help='Pixel size of the image, '
                               'defined by two floats. '
                               'First is length in X dimension, '
                               'then in Y dim.')
    optional.add_argument('--desired_pixel_size',
                          type=float,
                          default=(0.5, 0.5),
                          nargs='+',
                          help='Desired pixel size of the image, '
                               'defined by two floats. '
                               'First is length in X dimension, '
                               'then in Y dim.')
    optional.add_argument('--resize_tolerance',
                          default=0.05,
                          type=float,
                          help='Tolerance for pixel size, see `pixel_size` '
                               'argument.')
    optional.add_argument('--no_pixel_resize',
                          action='store_true',
                          help='Should image be resized to match usual pixel'
                               'size (~0.50um).')
    optional.add_argument('--preprocessings',
                          nargs='+',
                          help='Preprocessings used, currently supported:'
                               'denoising',
                          default=[])

    return parser


def save_output_slices_to_folder(output_path, output_slices):
    all_slices_path = os.path.join(output_path, '_segmented.tif')

    with imageio.get_writer(all_slices_path) as writer:
        for i, image_slice in enumerate(output_slices):
            output_path_slice = os.path.join(output_path,
                                             '{0:03}.tif'.format(i))
            imageio.imsave(output_path_slice, image_slice)

            writer.append_data(image_slice)


def preprocess(input_slices, preprocessings_list):
    """Transform `input_volume` according to `preprocessings_list`.

    Args:
        input_slices (Iterable)
        preprocessings_list (list): List of preprocessings (strings).

    Returns:
        Iterable: Volume with applied preprocessing.
    """
    for preprocessing in preprocessings_list:
        if preprocessing == 'denoising':
            input_slices = denoising_preprocess(input_slices)

    return input_slices


def denoising_preprocess(input_slices):
    """Denoise given `input_volume`.

    Args:
        input_slices (VolumeIter)

    Returns:
        VolumeIter: Volume with applied denoising.
    """
    model_path = 'models/denoising/model0.h5'
    model = load_model_with_cache(model_path)

    for input_slice in input_slices:
        denoised_slice = denoise.denoise_image(image=input_slice,
                                               model=model,
                                               batch_size=1,
                                               patches_shape=None,
                                               patches_stride=None)
        yield denoised_slice


def main(args):
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(args.output)

    with readers.read_one_channel_volume(args.input,
                                         channels=args.channel,
                                         series=args.series) as reader:
        input_volume = reader[args.start:args.stop]

        output_slices = \
            segmentation3d_by_2d(input_volume=input_volume,
                                 model=args.model,
                                 model_args=args.model_args,
                                 pixel_size=args.pixel_size,
                                 desired_pixel_size=args.desired_pixel_size,
                                 tolerance=args.resize_tolerance,
                                 no_pixel_resize=args.no_pixel_resize)

        if os.path.isdir(args.output):
            save_output_slices_to_folder(args.output, output_slices)
        else:
            imageio.mimwrite(args.output, output_slices)


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
