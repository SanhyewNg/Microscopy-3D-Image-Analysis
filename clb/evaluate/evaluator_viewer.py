import argparse
import glob
import os
import shutil

import imageio
import numpy as np
import skimage.color


def load_all_images(path):
    return sorted(glob.glob(os.path.join(path, '*.tif')) + glob.glob(os.path.join(path, '*.png')))


def parse_arguments():
    parser = argparse.ArgumentParser(description='CLB evaluation view preparer', add_help=True)
    parser.add_argument('evaluation_dir', help='path to evaluator output')
    parser.add_argument('output_dir', help='path to folder images will be copied for easier comparison')
    parser.add_argument('dirs_to_compare', metavar='N', type=str, nargs='+',
                        help='list of directories from which to take images')
    parser.add_argument('--rescale', dest='rescale', action='store_true', help='should rescale each image')
    parser.add_argument('--colour', help='list of one-based indexes of images to random label colour')

    parser.set_defaults(rescale=False)
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data from each directory.
    paths_to_compare = [(d, os.path.join(args.evaluation_dir, d)) for d in args.dirs_to_compare]
    all_images = [(d, load_all_images(p)) for d, p in paths_to_compare]

    # Check if all consist of the same set of files.
    dirs, image_lists = zip(*all_images)
    image_counts = list(map(len, image_lists))
    assert min(image_counts) == max(image_counts)

    images_to_colour = [] if args.colour is None else args.colour.split(",")

    # Save them with suffixes from dirs_to_compare.
    for i in range(0, image_counts[0]):
        for order, image_set in enumerate(all_images,1):
            current_dir = image_set[0].replace("\\", "_").replace("/","_")
            current_image_path = image_set[1][i]

            filename = os.path.basename(current_image_path)
            extension = os.path.splitext(filename)[1]
            appended_filename = "{0:03}".format(i+1) + "_" + str(order) + "_" + current_dir + "." + extension
            copied_path = os.path.join(args.output_dir, appended_filename)

            shutil.copy(current_image_path, copied_path)

            if args.rescale:
                image = imageio.imread(copied_path)
                # we want to make 8-bit image where the
                # non-zero pixels are more visible
                rescaled = image * float(230) / image.max()
                rescaled[rescaled != 0] += 25
                imageio.imwrite(copied_path, rescaled.astype(np.uint8))

            if str(order) in images_to_colour:
                image = imageio.imread(copied_path)
                coloured = skimage.color.label2rgb(image, bg_label=0) * 255
                imageio.imwrite(copied_path, coloured.astype(np.uint8))

            print("Copied ", appended_filename, "...")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
