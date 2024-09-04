import argparse
import os

import imageio
import numpy as np
import pandas

from clb.classify.visualisation import ClassificationVolume
from clb.image_processing import (find_corresponding_labels,
                                  remove_gaps_in_slices)
from clb.utils import bbox


def cell_level_from_contours(cell_labels, tissue_classes, overlap_threshold=0.5):
    """
    Uses tissue level annotation of prediction to assign class to the individual cells.
    Args:
        cell_labels: S x Y x X volume with cell labels.
        tissue_classes: S x Y x X volume with marked areas.
        overlap_threshold: required threshold on the class overlap fraction

    Returns:
        dictionary from cell label to its class
    """

    label_classes = find_corresponding_labels(cell_labels, tissue_classes, return_overlap=True, return_count=True)
    label_values = set(np.unique(cell_labels))
    label_values.remove(0)
    res = {}
    for i in label_values:
        res[i] = {"id": i}
        if i in label_classes:
            class_info = label_classes[i]
            class_value, fraction, count = class_info

            if fraction >= overlap_threshold:
                res[i]["class"] = class_value
            else:
                res[i]["class"] = 0

            res[i]["class_fraction"] = fraction
            res[i]["class_pixels"] = count
        else:
            res[i]["class"] = 0
            res[i]["class_fraction"] = 0
            res[i]["class_pixels"] = 0

    return res


def cell_level_classes(labels_volume, classes_volume, type, fill_gaps=False, overlap_threshold=0.5):
    """
    Determine cell level class using annotated or predicted class volume.
    Args:
        labels_volume: S x Y x X volume with cell labels.
        classes_volume: S x Y x X volume with marked areas or cell centers,
            may be only partially filled (one slice).
        type: 'tissue' or 'centers' depending on what type
            classes_volume is.
        fill_gaps: if there are gaps in labels, should they be filled from neighbouring
        overlap_threshold: required threshold on the class overlap fraction

    Returns:
        dictionary from cell label to its class
    """
    assert labels_volume.shape == classes_volume.shape

    # use only annotated area (e.g. only center slices is annotated)
    annotated_box = bbox(classes_volume, 0)
    if annotated_box is not None:
        first_annotated_z, last_annotated_z = annotated_box
    else:  # if none annotated then assume middle slice is
        first_annotated_z = last_annotated_z = (len(classes_volume)) // 2  # because of zero-index

    if fill_gaps:
        labels_volume = remove_gaps_in_slices(labels_volume)

    labels_volume = labels_volume[first_annotated_z:last_annotated_z + 1]
    classes_volume = classes_volume[first_annotated_z:last_annotated_z + 1]

    if type == "tissue":
        return cell_level_from_contours(labels_volume, classes_volume, overlap_threshold=overlap_threshold)
    else:
        raise Exception("Invalid classes_volume type.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='CLB labels to class matcher.', add_help=True)
    parser.add_argument('--labels', help='path to tiff volume with cell level labels', required=True)
    parser.add_argument('--classes', help='path to tiff volume with annotations', required=True)
    parser.add_argument('--type', help='whether annotation are "tissue" or cell "centers"', required=True)
    parser.add_argument('--output', help='output path to tiff volume with cell level classes', required=True)

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--output_tsv', help='output path to tsv with cell level classes')
    optional.add_argument('--fill_gaps', dest='fill_gaps', action='store_true', help="fill gaps in labels")
    parser.set_defaults(fill_gaps=False)

    parsed_args = parser.parse_args()
    return parsed_args


def main(args):
    labels_volume = imageio.volread(args.labels)
    classes_volume = ClassificationVolume.load(args.classes)

    label_classes = cell_level_classes(labels_volume, classes_volume, args.type, args.fill_gaps)

    # Prepare output tiff with matched classes.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Convert classes into 0-1 range so that it can be presented in overlay.
    id_to_class_1_2 = {k: v['class'] * 0.9 + 0.1 for k, v in label_classes.items()}
    overlay = ClassificationVolume.create(labels_volume, id_to_class_1_2)
    imageio.mimwrite(args.output, list(overlay))

    # If specified output tsv file
    if args.output_tsv:
        os.makedirs(os.path.dirname(args.output_tsv), exist_ok=True)
        df = pandas.DataFrame(list(label_classes.values()))
        df.to_csv(args.output_tsv, index=False, sep='\t')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
