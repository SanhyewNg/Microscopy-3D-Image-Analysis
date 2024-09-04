import argparse
import os

import imageio
import pandas
from tqdm import tqdm

import clb.run as run_instance
from clb.classify.cell_extractor import extract_all_cells_crops
from clb.classify.feature_extractor import extract_all_features
from clb.classify.extractors import parse_channels_preprocessing
from clb.classify.instance_matching import cell_level_classes
from clb.classify.utils import get_all_datasets, get_all_multiple_datasets, save_to_tsv, ClassVolumeSet
from clb.dataprep.utils import ensure_3d_rgb
from clb.image_processing import remove_annotated_blobs
from clb.yaml_utils import save_args_to_yaml
from clb.classify.visualisation import ClassificationVolume


def ensure_cell_labels(all_data, instance_model):
    for d in tqdm(all_data, "Ensure that instance segmentation exist"):
        if not d.label_exist:
            if not instance_model:
                raise Exception("No instance model to generate missing instance segmentation.")

            required_args = ['--input', d.input, '--outputs', d.label, '--model', instance_model, '--use_channel',
                             0]
            instance_args = run_instance.get_parser().parse_args(required_args)
            run_instance.main(instance_args)


def determine_cell_classes(all_data, fill_gaps=False, remove_blobs=False, overlap_threshold=0.5):
    for d in tqdm(all_data, "Determine class for each cell in instance segmentation"):
        labels_volume = imageio.volread(d.label)
        classes_volume = imageio.volread(d.gt)

        if remove_blobs:
            labels_volume = remove_annotated_blobs(labels_volume, labels_volume)

        d.cell_classes = cell_level_classes(labels_volume, classes_volume, "tissue", fill_gaps, overlap_threshold)

        assert d.cell_classes != {}, "No cells were matched to class for " + d.input


def extract_cell_predicted_classes(all_data):
    for d in tqdm(all_data, "Extract class for each cell in instance classification:"):
        labels_volume = imageio.volread(d.label)
        binary_prediction_classes_volume = ClassificationVolume.from_file(d.classes, binary=True)
        d.cell_classes_predicted = cell_level_classes(labels_volume, binary_prediction_classes_volume, "tissue")


def decide_voxel_size(data, voxel_resize_requested):
    if voxel_resize_requested:
        if data.voxel_size is None:
            raise Exception("Voxel resize requested but no voxel size data in training data.")
        return data.voxel_size
    return None


def extract_cell_features(all_data, channels, features_type, voxel_resize):
    for d in tqdm(all_data, "Extracting features for each cell in instance segmentation"):
        input_volume = imageio.volread(d.input)
        labels_volume = imageio.volread(d.label)

        # read yaml file and get data voxel size
        voxel_size = decide_voxel_size(d, voxel_resize)
        d.cell_features = extract_all_features(input_volume, labels_volume, channels,
                                               features_type, list(d.cell_classes.keys()),
                                               voxel_size=voxel_size)


def extract_cell_crops(all_data, channels, voxel_resize, crop_size):
    for d in tqdm(all_data, "Extracting crops for each cell in instance segmentation"):
        input_volume = imageio.volread(d.input)
        labels_volume = imageio.volread(d.label)

        # read yaml file and get data voxel size
        voxel_size = decide_voxel_size(d, voxel_resize)
        d.cell_crops = extract_all_cells_crops(input_volume, labels_volume, channels, crop_size,
                                               list(d.cell_classes.keys()), voxel_size=voxel_size)


def merge_all_features(all_data):
    training_data = {}
    for d in all_data:
        volume_name = d.crop_name
        volume_cell_data = d.merged_cell_data
        for id, data in volume_cell_data.items():
            training_data[(volume_name, id)] = data
    df = pandas.DataFrame.from_dict(training_data, orient='index')
    return df


def save_all_crops(all_data, output):
    negative_dir = os.path.join(output + "_cells", "0_negative")
    positive_dir = os.path.join(output + "_cells", "1_positive")
    uncertain_dir = os.path.join(output + "_cells", "2_uncentain")
    class_dirs = [negative_dir, positive_dir, uncertain_dir]

    for class_dir in class_dirs:
        os.makedirs(class_dir, exist_ok=True)

    all_cells = [(d.crop_name, d.cell_classes, cell_id, crop) for d in all_data for cell_id, crop in
                 d.cell_crops.items()]

    for name, cell_classes, cell_id, crop in tqdm(all_cells, "Saving extracted crops"):
        cell_type = cell_classes[cell_id]['class']
        cell_dir = class_dirs[cell_type]

        output_name = "{0}.{1}.input.tif".format(name, cell_id)
        imageio.mimwrite(os.path.join(cell_dir, output_name), ensure_3d_rgb(crop["raw_input"]))

        output_name = "{0}.{1}.contour.tif".format(name, cell_id)
        imageio.mimwrite(os.path.join(cell_dir, output_name), crop["raw_contour"])


def process_volume_sets(volume_sets, instance_model, channels_preprocessings_list, features_type, fill_gaps=False,
                        remove_blobs=False, voxel_resize=False, overlap_threshold=0.5, cell_crop_size=None):
    """
    For given data sets prepares their classification data. If necessary calculates their instance segmentation.
    Args:
        volume_sets: list of ClassVolumeSet which are to be processed
        instance_model: path to h5 file of the instance segmentation model
        channels_preprocessings_list: list of channels (with potential preprocessings) for which we want to calculate features
        features_type: type of the features to calculate
                       or None to skip features calculation
        fill_gaps: fill the gaps in the instance segmentation (designed to work with manual annotation)
        remove_blobs: remove instances annotated as blobs (designed to work with manual annotation)
        voxel_resize: if True then for feature extraction every dataset should be resized to comply with standard voxel size
        overlap_threshold: required threshold on the class overlap fraction
        cell_crop_size: if not None then crops of specified size in um are extracted for each cell

    Returns:
        processed list of ClassVolumeSet
    """
    # Ensure that instance segmentation is available
    ensure_cell_labels(volume_sets, instance_model)

    # Match classes for every cell in every volume.
    determine_cell_classes(volume_sets, fill_gaps, remove_blobs, overlap_threshold)

    if features_type is not None:
        # Calculate features for every cell in every volume
        extract_cell_features(volume_sets, channels_preprocessings_list, features_type, voxel_resize)

    if cell_crop_size is not None:
        extract_cell_crops(volume_sets, channels_preprocessings_list, voxel_resize, cell_crop_size)

    return volume_sets


def parse_arguments():
    parser = argparse.ArgumentParser(description='CLB class training data preparer.', add_help=True)
    required = parser.add_argument_group('required arguments')

    required.add_argument('--data', help='path to directory with train, val and test dirs, '
                                         'required if --annotated_input or --annotated_gt not specified')
    required.add_argument('--datasets',
                          help='used and required if --data is specified, describes which datasets should be used:'
                               'e.g. "train+val", "test", "val+test"')

    required.add_argument('--annotated_input', help='path to folder with input imagery')
    required.add_argument('--annotated_gt', help='path to folder with class annotations')

    required.add_argument('--channels', help='channels with optional preprocessings to use e.g. "1,2" or "1-equal,1"',
                          required=True)
    required.add_argument('--labels', help='path to folder with cell labels', required=True)
    required.add_argument('--output', help='output path to tsv with cell features and classes', required=True)

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--instance_model', help='path to h5 model for instance segmentation')
    optional.add_argument('--class_name', help='name of the class to predict used to select annotations')
    optional.add_argument('--features_type', help='identifier of the feature set to extract', default="default")
    optional.add_argument('--cell_crop_size', type=float, default=None,
                          help='if specified crops for each cell will be extracted of specified size in um')
    optional.add_argument('--no_voxel_resize', action='store_true',
                          help='should volume be resized to match usual pixel size.')
    parser.set_defaults(no_voxel_resize=False)
    optional.add_argument('--overlap_threshold', type=float,
                          help='required threshold on the class overlap fraction', default=0.5)

    parsed_args = parser.parse_args()
    if (parsed_args.data is None or parsed_args.datasets is None) and \
            (parsed_args.annotated_input is None or parsed_args.annotated_gt is None):
        raise Exception(
            "Either '--data' and '--datasets' or both '--annotated_input' and '--annotated_gt' has to be provided.")

    return parsed_args


def main(args):
    if args.data is not None and args.datasets is not None:
        datasets_list = args.datasets.split("+")
        all_data = get_all_multiple_datasets(args.data, datasets_list, "input", "annotations",
                                             args.labels, None, args.class_name)
    else:
        all_data = get_all_datasets(args.annotated_input, args.annotated_gt,
                                    args.labels, None, args.class_name)

    print("Loaded data:")
    for d in all_data:
        print("    Data name: " + os.path.basename(d.input))
        print("        " + str(d))

    # Save params next to the resulting features.
    output_dir = os.path.dirname(args.output)
    output_filename = os.path.splitext(os.path.basename(args.output))[0]

    os.makedirs(output_dir, exist_ok=True)
    save_args_to_yaml(args.output, args)

    channels = parse_channels_preprocessing(args.channels)
    process_volume_sets(all_data, args.instance_model, channels, args.features_type,
                        voxel_resize=not args.no_voxel_resize,
                        overlap_threshold=args.overlap_threshold,
                        cell_crop_size=args.cell_crop_size)

    # Save crops for each cell if specified.
    if args.cell_crop_size is not None:
        save_all_crops(all_data, os.path.join(output_dir, output_filename))

    # Merge data from each volume into one tsv file.
    df = merge_all_features(all_data)
    save_to_tsv(df, args.output)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
