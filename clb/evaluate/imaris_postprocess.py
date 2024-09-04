import argparse
import functools

import imageio
import numpy as np
import skimage

from clb.segment.segment_cells import label_cells_by_layers, dilation_only_2d


def relabel_cells(v):
    relabeled = skimage.measure.label(v)
    return relabeled


def parse_arguments():
    parser = argparse.ArgumentParser(description='CLB postprocess of IMARIS results.', add_help=True)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--input', help='path to tiff file with IMARIS labels', required=True)
    required.add_argument('--output', help='path to processed output file', required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--relabel', help='expected Y x X size of the output ', dest='relabel',
                          action='store_true')
    optional.add_argument('--dilation', help='expected Y x X size of the output ', type=int, default=3)
    optional.add_argument('--layer_split_size', help='Z splits in processing ', type=int, default=20)

    parser.set_defaults(relabel=False)

    parsed_args = parser.parse_args()
    return parsed_args


def main(args):
    labels_volume = imageio.volread(args.input).astype(np.uint32)

    if args.relabel:
        labels_volume = label_cells_by_layers(labels_volume, relabel_cells, args.layer_split_size)

    if args.dilation:
        labels_volume = label_cells_by_layers(labels_volume,
                                              functools.partial(dilation_only_2d, dilation=args.dilation),
                                              args.layer_split_size)

    imageio.mimwrite(args.output, list(labels_volume))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
