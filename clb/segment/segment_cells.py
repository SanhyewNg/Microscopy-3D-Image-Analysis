import argparse
import os

import imageio
import numpy as np
import scipy
import scipy.ndimage
from tqdm import tqdm

from clb.dataprep.imaris.ims_file import ImsFile
from clb.dataprep.utils import load_tiff_stack, rescale_to_float
from clb.image_processing import (find_corresponding_labels,
                                  restrict_struct_to_only_2d)
from clb.segment.instance.components import connected_components
from clb.segment.instance.watersheds import distance_watershed
from clb.utils import (replace_values,
                       replace_values_in_slices)
from clb.volume_slicer import VolumeSlicer
from clb.yaml_utils import merge_yaml_to_cli, save_args_to_yaml


def make_consistent_labels(volumes1, volume2, return_copy=True):
    """
    Remap labels in volumes1 and volume2 so that they are consistent - same
    value for same objects in volumes1 and volumes2. It assumes that last
    image in volumes1 and first image in volume2 represent the same objects.
    It shifts the rest of the volume2 labels up. Values in volumes1 can
    change if they are one object in volume2.
    Args:
        volumes1: S1 x Y x X or a list of S1 x Y x X sorted so that volume
                  touching volume2 is last
        volume2: S2 x Y x X
        return_copy: should return copy of volume1 and volume2 or should make
                     changes in place

    Returns:
        volume1 with values potentially merged
        volume2 with values remaped so they are consistent with volume1
    """
    volume2_changed = volume2.copy() if return_copy else volume2
    volumes1_changed = volumes1.copy() if return_copy else volumes1
    volume1_last_changed = volumes1_changed[-1] \
        if isinstance(volumes1_changed, list) else volumes1_changed

    volume2_changed[volume2_changed > 0] += volume1_last_changed.max()
    mapping = find_corresponding_labels(volume2_changed[0],
                                        volume1_last_changed[-1])
    volume2_mapped = replace_values(volume2_changed, mapping, False)

    mapping2 = find_corresponding_labels(volume1_last_changed[-1],
                                         volume2_mapped[0])

    replace_values_in_slices(volumes1_changed[::-1], mapping2)

    return volumes1_changed, volume2_mapped


def dilation_only_2d(labels, dilation):
    """
    GDilate the labels in numpy array but only across Y x X layers.

    Args:
        labels: S x Y x X numpy array of labels
        dilation: number of iterations of 3x3 structure

    Returns:
        volume with labeled enlarged
    """

    if dilation > 0:
        struct = scipy.ndimage.morphology.generate_binary_structure(3, 1)
        footprint = scipy.ndimage.morphology.iterate_structure(struct, dilation)
        # We do not want to dilate in Z direction.
        footprint = restrict_struct_to_only_2d(footprint)
        labels = scipy.ndimage.morphology.grey_dilation(labels,
                                                        footprint=footprint)
    return labels


def label_cells_watershed(input_volume, threshold, use_intensity=False,
                          median_filtering=3, peak_suppress=5, smoothing=2.0,
                          z_step_ratio=0.2, dilation=2):
    """
    Use watershed on distance transform combined with intensity of input volume
    to convert intensities in input volume into cell labels.

    Args:
        input_volume: S x Y x X numpy array
        threshold: 0-1 threshold
        use_intensity: should intensity be used in watershed
        median_filtering: filtering of the thresholded mask
        peak_suppress: minimum distance between peaks
        smoothing: smoothing of the distance image before peaks are calculated
        z_step_ratio: how much dimension in x or y is related to z. If z slices
            are sparser the ratio should be smaller.
        dilation: number of iterations of dilation of the resulting labels

    Returns:
        volume with cells labeled with uint32 numbers
    """
    input_volume = rescale_to_float(input_volume, float_type='float32')

    object_mask = input_volume > threshold
    # closing was tested here but yielded worse results
    cleaned_mask = scipy.ndimage.median_filter(object_mask, median_filtering)
    intensity_volume = input_volume if use_intensity else None
    labels = distance_watershed(cleaned_mask, intensity=intensity_volume,
                                suppress=peak_suppress, smooth=smoothing,
                                z_ratio=z_step_ratio)

    if dilation > 0:
        labels = dilation_only_2d(labels, dilation)

    return labels


def label_cells_cc(input_volume, threshold, opening, dilation, small_thresh=20):
    """
    Using global thresholding and connected component to convert intensities
    in input volume into cell labels.

    Args:
        input_volume: S x Y x X numpy array
        threshold: 0-1 threshold
        opening: number of morphology opening applied after threshold
        dilation: number of iterations of dilation of the resulting labels
        small_thresh: remove objects smaller than

    Returns:
        volume with cells labeled with uint32 numbers
    """
    input_volume = rescale_to_float(input_volume, float_type='float32')
    labels = connected_components(input_volume, input_volume,
                                  np.zeros(input_volume.shape),
                                  opening=opening,
                                  obj_threshold=threshold,
                                  small_obj_thresh=small_thresh)[1]
    if dilation > 0:
        labels = dilation_only_2d(labels, dilation)

    return labels


def label_cells_by_layers(input_volume, label_layer, layers_per_one):
    """
    Post-process volume using VolumeSlicer to split it into a number of
    thinner slices. The labels from neighbouring slices are made consistent
    using the overlapping image.

    Args:
        input_volume: S x Y x X numpy array
        label_layer: method used to post-process one thick slice, should
                     return uint32 array
        layers_per_one: number of layers in one thick slice

    Returns:
        post-process volume is uint32
    """
    slicer = VolumeSlicer(layers_per_one)
    volume_parts = slicer.divide_volume(input_volume)
    res = []
    for part in tqdm(volume_parts, disable=len(volume_parts) <= 1, desc="Segmenting"):
        new_slices = label_layer(part).astype(np.uint32)
        # Merge labels if already exists
        if res:
            make_consistent_labels(res, new_slices, return_copy=False)
        res.append(new_slices)

    return slicer.stitch_volume(res)


def get_parser():
    parser = argparse.ArgumentParser(
        description='CLB find cells in probability map 3D.',
        add_help=False)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--input', help='3d probability tiff path',
                          required=True)
    required.add_argument('--output', help='output path', required=True)

    optional = parser.add_argument_group('optional arguments')

    optional.add_argument('-h', '--help', action="help",
                          help='show this help message and exit')

    optional.add_argument('--method',
                          help=('method used for instance segmentation: '
                                'components or watershed'))

    optional.add_argument('--layer_split_size', type=int,
                          help='maximum height of the layers batch for '
                               + 'identification', default=17)

    optional.add_argument('--start_slice', type=int, default=0)

    optional.add_argument('--last_slice', type=int, default=10000)

    optional.add_argument('--saved_args', type=str,
                          help='path to file with parameters')

    optional.add_argument('--threshold', type=float, help='threshold in [0-1]')

    optional.add_argument('--dilation', type=int, default=2,
                          help='number of iterations of dilation of resulting '
                               'labels to compensate')

    components = parser.add_argument_group('arguments for method=components')

    components.add_argument('--opening', type=int, default=1,
                            help='0,1,2 - no, 3x3, 5x5 gray opening')

    watershed = parser.add_argument_group('arguments for method=watershed')
    watershed.add_argument('--smooth_mask', type=int,
                           help=('median filtering '
                                 'of thresholded mask (1,3,5,7)'),
                           default=3)
    watershed.add_argument('--suppress_peaks', type=int,
                           help='how far should cells be (1,2,3,4,5...)',
                           default=5)

    watershed.add_argument('--smooth_distances', type=float,
                           help='smoothing factor of distance [0-5]',
                           default=2)
    watershed.add_argument('--use_intensity', type=bool,
                           help='use prediction intensity',
                           default=False)
    watershed.add_argument('--z_step', type=float, help='z_step ratio',
                           default=0.2)

    return parser


def parse_arguments():
    return get_parser().parse_args()


def merge_with_saved(args, path_to_saved):
    params_to_merge_general = ['method', 'threshold', 'dilation']
    params_to_merge_components = ['opening']
    params_to_merge_watershed = ['smooth_mask', 'suppress_peaks',
                                 'smooth_distances', 'use_intensity', 'z_step']
    return merge_yaml_to_cli(path_to_saved, vars(args),
                             params_to_merge_general +
                             params_to_merge_components +
                             params_to_merge_watershed)


def main(args, ims_file=None):
    if args.saved_args is not None:
        args = merge_with_saved(args, args.saved_args)

    save_args_to_yaml(os.path.splitext(args.output)[0] + '_ident', args)

    def label_cc(v):
        return label_cells_cc(v, args.threshold, args.opening, args.dilation)

    def label_watershed(v):
        return label_cells_watershed(v, args.threshold, args.use_intensity,
                                     args.smooth_mask, args.suppress_peaks,
                                     args.smooth_distances, args.z_step,
                                     args.dilation)

    def get_label_method():
        if args.method == 'watershed':
            return label_watershed
        else:
            return label_cc

    volume = load_tiff_stack(args.input)
    volume[0:args.start_slice] = 0
    volume[args.last_slice + 1:] = 0

    labels = label_cells_by_layers(volume, get_label_method(),
                                   args.layer_split_size)

    #  Let's keep 16-bit so they are strictly unique.
    #  Make sure that it is saved as multipage.

    file_format = os.path.splitext(args.output)[1]
    if 'tif' in file_format:
        imageio.mimwrite(args.output, list(labels))
    elif file_format == '.ims':
        output_arr = np.asarray(list(labels))

        if ims_file is None:
            file = ImsFile(args.output, mode='x')
            file.add_channel(output_arr, color_mode='TableColor',
                             channel_name='segmentation_labels')
        else:
            ims_file.add_channel(output_arr, color_mode='TableColor',
                                 channel_name='DAPI_labels')


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
