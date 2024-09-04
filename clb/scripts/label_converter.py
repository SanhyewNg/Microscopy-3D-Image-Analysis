import argparse

import imageio
import numpy as np
import skimage.color
import skimage.segmentation


def to_8bit(v):
    return (v % 256).astype(np.uint8)


def to_colour(v):
    res = np.zeros(list(v.shape) + [3], dtype=np.uint8)
    for i, slice in enumerate(v):
        res[i] = (skimage.color.label2rgb(slice, bg_label=0) * 255).astype(np.uint8)
    return res


def separate(v):
    borders = skimage.segmentation.find_boundaries(v, mode='outer')
    v = v * (borders == 0)
    return v


def to_binary(v):
    return (v > 0).astype(v.dtype)


def parse_arguments():
    parser = argparse.ArgumentParser(description='CLB utility converter of labels.', add_help=True)
    parser.add_argument('input', help='path to tiff volume with labels')
    parser.add_argument('output', help='path to processed output file')
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--to_8bit', help='convert ensuring some separation ', dest='to_8bit',
                          action='store_true')
    optional.add_argument('--to_colour', help='convert labels to colours ', dest='to_colour',
                          action='store_true')
    optional.add_argument('--separate', help='separate labels', dest='separate',
                          action='store_true')
    optional.add_argument('--to_binary', help='make binary', dest='to_binary',
                          action='store_true')

    parsed_args = parser.parse_args()
    return parsed_args


def main(args):
    labels_volume = imageio.volread(args.input)

    if args.separate:
        labels_volume = separate(labels_volume)

    if args.to_8bit:
        labels_volume = to_8bit(labels_volume)

    if args.to_colour:
        labels_volume = to_colour(labels_volume)

    if args.to_binary:
        labels_volume = to_binary(labels_volume)

    imageio.mimwrite(args.output, list(labels_volume))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
