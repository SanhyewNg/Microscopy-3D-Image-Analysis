import argparse
import functools as ft
import os
import shutil
import warnings

import imageio
import numpy as np
from tqdm import tqdm

import clb.dataprep.imaris.ims_file as imsfile
import clb.dataprep.readers as readers
import clb.dataprep.uff.uff_writers as uwriters
import clb.dataprep.utils as utils
import clb.predict.predict3d as predict3d
import clb.segment.segment_cells as segment
import clb.yaml_utils as yutils


class SegmentationMethodError(Exception):
    """Raised when unexpected segmentation method is passed in arguments."""


def make_watershed_method(args):
    """Create function that performs watershed on given volume.

    Args:
        args (argparse.Namespace): Arguments for identification.

    Returns:
        Function that takes volume with probabilities and returns volume with
        labels.

    Raises:
        SegmentationMethodError: If `label_method` in args is not watershed.
    """
    if not args.method == 'watershed':
        raise SegmentationMethodError('Segmentation method is not watershed.')

    label_method = ft.partial(segment.label_cells_watershed,
                              threshold=args.threshold,
                              use_intensity=args.use_intensity,
                              median_filtering=args.smooth_mask,
                              peak_suppress=args.suppress_peaks,
                              smoothing=args.smooth_distances,
                              z_step_ratio=args.z_step,
                              dilation=args.dilation)

    return label_method


def make_segmentation(volume, args):
    """Use watershed to find cell labels.

    Args:
        volume (np.ndarray): Volume with prediction.
        args (argparse.Namespace): Postprocess arguments.

    Returns:
        np.ndarray: Volume with labels.
    """
    label_method = make_watershed_method(args)

    merge_labels = ft.partial(segment.label_cells_by_layers,
                              label_layer=label_method,
                              layers_per_one=30)
    labels = merge_labels(volume)

    return labels


def postprocess(output_slices, args):
    """Create labels from `output_slices` (prediction).

    If image was resized it is resized back to its size and then labels are
    calculated.

    Args:
        output_slices (Iterable): Slices after prediction.
        args (argparse.Namespace): Postprocess arguments.

    Returns:
        np.ndarray: Volume with labels, shape (Z, Y, X).
    """
    array_of_slices = np.stack(tqdm(output_slices, "Predicting"))
    labels = make_segmentation(array_of_slices, args)

    return labels


def write_imaris(path, input_volume, labels):
    """Write `input_volume` and `labels` to imaris file at `path`.

    Args:
        path (str): Path to save imaris file.
        input_volume (array_like): Input volume.
        labels (array_like): Volume with cell labels.
    """
    with imsfile.ImsFile(path,
                         mode='x',
                         image_metadata=input_volume.metadata) as ims_file:
        ims_file.add_channel(np.squeeze(input_volume),
                             color_mode='BaseColor',
                             color_value='Blue',
                             channel_name='DAPI')
        ims_file.add_channel(np.squeeze(labels),
                             color_mode='TableColor',
                             channel_name='DAPI_labels')


def write_uff(path, metadata, labels):
    """Save segmentation results to .uff.

    Args:
        path (str): Path to output file.
        metadata (dict)
        labels (np.ndarray): Volume with cell labels.
    """
    if labels.ndim < 4:
        labels = labels[..., np.newaxis]
        (metadata['SizeZ'], metadata['SizeY'], metadata['SizeX'],
         metadata['SizeC']) = map(str, labels.shape)      

    assert str(metadata['SizeC']) == '1', "Only segmentation channel should be present."
    metadata['Channels'] = [{'Name': 'segmentation'}]

    if metadata.get("Name") is None:
        metadata['Name'] = os.path.basename(path)  

    writer = uwriters.UFFWriter(path, data=labels, metadata=metadata)
    writer.write()


def get_parser():
    parser = argparse.ArgumentParser(description='CLB instance segmentation.',
                                     add_help=True)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--input',
                          help='Path to .tif or .lif input file.',
                          required=True)
    required.add_argument('--outputs',
                          help='Paths to save outputs to. Filenames should '
                               'end with .tif or .ims. Every pair of curly '
                               'brackets in path will be replaced with '
                               'series number. If directories does not '
                               'exist, they will be created.',
                          required=True,
                          nargs='+')
    required.add_argument('--model',
                          help=('Path to h5 model (expects also the existence '
                                'of corresponding yaml files).'),
                          required=True)

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--use_channel',
                          type=int,
                          default=None,
                          help='Use specific channel from input. If None all '
                               'channels will be read and then max will be '
                               'taken from them.')
    optional.add_argument('--series',
                          type=int,
                          default=None,
                          help='Which series to segment (.lif files).')
    optional.add_argument('--start',
                          type=int,
                          default=0,
                          help='Starting slice for segmentation.')
    optional.add_argument('--stop',
                          type=int,
                          default=None,
                          help='Last slice for segmentation. It is not '
                               'segmented. If not given, all '
                               'slices from the starting slice are segmented.')
    optional.add_argument('--no_pixel_resize',
                          action='store_true',
                          help='Should image be resized to match usual pixel'
                               'size.')
    optional.add_argument('--pixel_size',
                          nargs='+',
                          type=float,
                          default=(0.5, 0.5),
                          help='Pixel size of the image, '
                               'defined by two floats. '
                               'First is length in X dimension, '
                               'then in Y dim.')
    optional.add_argument('--desired_pixel_size',
                          nargs='+',
                          type=float,
                          default=(0.5, 0.5),
                          help='Pixel size of the image, '
                               'defined by two floats. '
                               'First is length in X dimension, '
                               'then in Y dim.')
    optional.add_argument('--resize_tolerance',
                          default=0.05,
                          type=float,
                          help='Tolerance for pixel size, see `pixel_size` '
                               'argument.')
    return parser


def make_instance_segmentation(input,
                               model,
                               outputs,
                               series,
                               ident_args=None,
                               start=0,
                               stop=None,
                               pixel_size=(0.5, 0.5),
                               desired_pixel_size=(0.5, 0.5),
                               use_channel=None,
                               resize_tolerance=0.05,
                               no_pixel_resize=False):
    """Perform instance segmentation on specified series.

    See parser description for information about arguments.
    """
    assert model is not None, "Instance model was not given."

    with readers.read_one_channel_volume(input, channels=use_channel,
                                         series=series) as reader:
        input_volume = reader[start:stop]
        
        for path in outputs:
            utils.ensure_path(path, extensions=('.tif', '.ims'))

        model_args = model.replace('.h5', '.yaml')
        output_slices = \
            predict3d.segmentation3d_by_2d(input_volume=input_volume,
                                           model=model,
                                           model_args=model_args,
                                           tolerance=resize_tolerance,
                                           pixel_size=pixel_size,
                                           desired_pixel_size=desired_pixel_size,
                                           no_pixel_resize=no_pixel_resize)

        if ident_args is None:
            ident_args_path = model.replace('.h5', '_ident.yaml')
            ident_args = yutils.load_yaml_args(ident_args_path)

        labels = postprocess(output_slices, ident_args)

        for path in outputs:
            if path.endswith('.tif'):
                imageio.mimwrite(path, labels)
            elif path.endswith('.ims'):
                write_imaris(path,
                             input_volume,
                             labels)
            else:
                write_uff(path, input_volume.metadata, labels)


def main(args):
    """Perform instance segmentation on one series.

    See parser description for information about arguments.
    """
    if args.series is not None and not args.input.endswith('.lif'):
        warnings.warn(
            'Used series argument with .tif file, it will be ignored.')
        args.series = 0

    metadata = readers.get_metadata(path=args.input, series=args.series)
    name = metadata.get('Name', 'series_{}'.format(args.series))
    outputs = [path.format(name=name) for path in args.outputs]

    ident_args = None
    for path in outputs:
        utils.ensure_path(path, extensions=('.tif', '.ims'))
        output_dir, basename = os.path.split(path)
        basename = os.path.splitext(basename)[0]

        args_path = os.path.join(output_dir, basename + '_args')
        yutils.save_args_to_yaml(args_path, args)

        saved_iargs_path = args.model.replace('.h5', '_ident.yaml')
        ident_args = yutils.load_yaml_args(saved_iargs_path)
        ident_args_path = os.path.join(output_dir,
                                       basename + '_args_ident.yaml')
        shutil.copy(src=saved_iargs_path, dst=ident_args_path)

    make_instance_segmentation(input=args.input,
                               model=args.model,
                               ident_args=ident_args,
                               series=args.series,
                               outputs=outputs,
                               start=args.start,
                               stop=args.stop,
                               pixel_size=args.pixel_size,
                               desired_pixel_size=args.desired_pixel_size,
                               use_channel=args.use_channel,
                               resize_tolerance=args.resize_tolerance,
                               no_pixel_resize=args.no_pixel_resize)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
