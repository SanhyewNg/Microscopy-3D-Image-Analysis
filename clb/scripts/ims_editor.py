import argparse

import numpy as np

from clb.dataprep.imaris.ims_file import ImsFile
from clb.dataprep.utils import load_tiff_stack


def get_parser():
    parser = argparse.ArgumentParser(description='IMS file export script.', add_help=True)
    parser.add_argument('--dapi', help='path to tiff volume with dapi', required=True)
    parser.add_argument('--ki', help='path to tiff volume with ki')
    parser.add_argument('--mem', help='path to tiff volume with mem')
    parser.add_argument('--labels', help='path to tiff volume with labels')
    parser.add_argument('--probs', help='path to tiff volume with probs')
    parser.add_argument('--probs_name', help='name of the channel with probability')

    parser.add_argument('--output', help='output path to ims file', required=True)

    return parser


def add_basecolor_channel_from_file(ims_file, tiff_path, color_value, name):
    ims_file.add_channel(load_tiff_stack(tiff_path),
                         color_mode='BaseColor',
                         color_value=color_value,
                         channel_name=name)


def main(args):
    f = ImsFile(args.output, mode='x')
    if args.dapi:
        print("DAPI...")
        add_basecolor_channel_from_file(f, args.dapi, 'Blue', 'DAPI')

    if args.ki:
        print("Ki67...")
        add_basecolor_channel_from_file(f, args.ki, 'Yellow', 'Ki67')

    if args.mem:
        print("PanCK...")
        add_basecolor_channel_from_file(f, args.mem, 'Green', 'PanCK')

    if args.labels:
        print("DAPI_labels...")
        f.add_channel(load_tiff_stack(args.labels),
                      color_mode='TableColor',
                      channel_name='DAPI_labels')

    if args.prob:
        name = args.prob_name or 'CellClass'
        print(name + "...")
        add_basecolor_channel_from_file(f, args.probs, 'Red', name)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
