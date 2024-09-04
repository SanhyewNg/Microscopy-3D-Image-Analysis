import argparse
import glob
import os
import shutil
import subprocess
import tempfile

import imageio
import keras
from tqdm import tqdm

import clb.predict.predict3d as predict3d
import clb.segment.segment_cells as segment_cells
from clb.cropping import CropInfo
from clb.dataprep.utils import load_tiff_stack
from clb.image_processing import (remove_annotated_blobs,
                                  remove_annotated_blobs_when_overlap)
from clb.predict.predict_tile import load_model_with_cache

FLATTEN_PROB_SUFIX = '_prob'
FLATTEN_LABELS_SUFIX = '_labels'


class VolumeSet:
    def __init__(self, input, crop_info, gt, prob, label, existing_label_volume=None):
        """
        Prepare set of paths which represent: input, annotation, probabilities and labels for one annotated volume. 
        If provided CropInfo and labeled volume it can crop labeled volume to get label crop. 
        Args:
            input: path to file with input image
            crop_info: path to yaml file with crop info
            gt: path to file with annotation
            prob: path to file with probability image
            label: path to file with labels
            existing_label_volume: optional one-channel 3d numpy array with labels
        """
        self.input = input
        self.gt = gt
        self.crop_info = crop_info
        self.prob = prob
        self.label = label
        self.prob_exist = os.path.isfile(prob)
        self.label_exist = label is not None and os.path.isfile(label) or existing_label_volume is not None
        self.existing_label_volume = existing_label_volume

    def load_gt(self):
        """
        Load gt from file and clean the blob areas.
        Returns:
            list of Y x X numpy arrays with blob annotations removed
        """
        gt = load_tiff_stack(self.gt)
        return [remove_annotated_blobs(l, l) for l in gt]

    def load_labels(self, minimum_blob_overlap=0.6):
        """
        Load label image from file or crop the existing_label_volume. 
        Labels in blob areas are then removed.
        Args:
            minimum_blob_overlap: minimum intersection / object to match object with blob
                (only if blob is the most overlaping label)
        Returns:
             list of Y x X numpy arrays with blob areas removed
        """
        gt_raw_labels = load_tiff_stack(self.gt)

        if self.existing_label_volume is not None:
            crop_infos = self.load_crop_info()
            loaded_labels = [info.crop(self.existing_label_volume) for info in crop_infos]
        else:
            loaded_labels = list(load_tiff_stack(self.label))

        return [remove_annotated_blobs_when_overlap(l, g, minimum_blob_overlap=minimum_blob_overlap) for l, g in
                zip(loaded_labels, gt_raw_labels)]

    def load_crop_info(self):
        if self.crop_info is None:
            return None
        else:
            return CropInfo.load(self.crop_info)

    def __str__(self):
        sb = []
        for key in ['input', 'crop_info', 'gt', 'prob', 'label']:
            sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        return ', '.join(sb)


def get_all_datasets(input, ground_truth, probability, labels):
    os.makedirs(input, exist_ok=True)
    os.makedirs(ground_truth, exist_ok=True)
    os.makedirs(probability, exist_ok=True)
    # Handle case when labels is actually a file where annotated data has to be cut from.
    crop_labels = os.path.isfile(labels)
    labels_volume = None
    if not crop_labels:
        os.makedirs(labels, exist_ok=True)
    else:
        labels_volume = load_tiff_stack(labels)

    input_files = sorted(glob.glob(os.path.join(input, '*.tif')))
    gt_files = sorted(glob.glob(os.path.join(ground_truth, '*.tif')))

    res = []
    for input, gt in zip(input_files, gt_files):
        file_name = os.path.basename(input)
        file_name_without_extension = file_name[:-4]
        input_path_without_extension = input[:-4]

        expected_crop_info = input_path_without_extension + ".yaml"
        expected_prob = os.path.join(probability, file_name_without_extension + "_prob.tif")

        # Set path to label only if not cropping from volume.
        if not crop_labels:
            expected_label = os.path.join(labels, file_name_without_extension + "_labels.tif")
        else:
            expected_label = None

        dataset = VolumeSet(input, expected_crop_info, gt, expected_prob, expected_label,
                            existing_label_volume=labels_volume)
        res.append(dataset)
    return res


def ensure_probabilites(all_data, args):
    for d in tqdm(all_data, "Ensure that predictions are not necessary or exist"):
        if (not d.prob_exist or args.regen_prob) and (not d.label_exist or args.regen_labels):
            if args.model is None:
                raise Exception("Predictions need to be calculated but no network model provided ")

            arguments = [
                '--input', d.input,
                '--output', d.prob,
                '--trim_method', 'padding',
                '--model', args.model,
                '--model_args', None,
            ]

            prediction_args = predict3d.get_parser().parse_args(arguments)

            expected_yaml_path = os.path.splitext(args.model)[0] + ".yaml"
            if os.path.isfile(expected_yaml_path):
                prediction_args.model_args = expected_yaml_path

            predict3d.main(prediction_args)


def ensure_labels(all_data, args):
    for d in tqdm(all_data, "Ensure that labels exist"):
        if not d.label_exist or args.regen_labels:
            required_args = ['--input', d.prob, '--output', d.label]
            identify_args = segment_cells.get_parser().parse_args(required_args)
            # Do not split or skip any slices.
            identify_args.layer_split_size = 30
            identify_args.start_slice = 0
            identify_args.last_slice = 10000

            # Sensible defaults.
            identify_args.method = 'components'
            identify_args.threshold = 0.85
            identify_args.opening = 1
            identify_args.dilation = 2

            identify_args.saved_args = None
            expected_yaml_path = None
            if args.model is not None:
                expected_yaml_path = os.path.splitext(args.model)[0] + "_ident.yaml"
            if args.identify_yaml is not None:
                expected_yaml_path = args.identify_yaml

            if expected_yaml_path is not None and os.path.isfile(expected_yaml_path):
                identify_args.saved_args = expected_yaml_path

            segment_cells.main(identify_args)


def flatten_into_folder(paths_or_volumes, output_folder, save_mapping=False):
    """
    Extract single images from multipage files or crop them from preloaded volumes.
    Save them as separate files.
    Args:
        paths_or_volumes: list of paths or list of volume to extract single images from.
        output_folder: folder where to put images
        save_mapping: save image number to filename to file
    """
    os.makedirs(output_folder, exist_ok=True)
    mapping = {}
    idx = 1

    for path_or_volume in paths_or_volumes:
        is_path = isinstance(path_or_volume, str)
        if is_path:
            volume = load_tiff_stack(path_or_volume)
        else:
            volume = path_or_volume

        for page, tiff in enumerate(volume, 1):
            tiff_path = os.path.join(output_folder, '{0:03}.tif'.format(idx))
            imageio.imsave(tiff_path, tiff)
            if save_mapping and is_path:
                mapping[idx] = path_or_volume + "_" + str(page)
            idx += 1

    if save_mapping:
        with open(os.path.join(output_folder, "mapping.txt"), 'w') as f:
            for l in mapping.items():
                print(l, file=f)


def flatten_all_data(all_data, minimum_blob_overlap, result_folder_prefix, output_path):
    flat_input = os.path.join(output_path, "data_input")
    flat_annotation = os.path.join(output_path, "data_annotation")
    flat_prob = os.path.join(output_path, result_folder_prefix + FLATTEN_PROB_SUFIX)
    flat_labels = os.path.join(output_path, result_folder_prefix + FLATTEN_LABELS_SUFIX)

    flatten_into_folder([d.input for d in all_data], flat_input, save_mapping=True)
    flatten_into_folder([d.load_gt() for d in all_data], flat_annotation)
    # Ignore probability if some propability images missing (labels were provided instead).
    if all([os.path.isfile(d.prob) for d in all_data]):
        flatten_into_folder([d.prob for d in all_data], flat_prob)
    else:
        print("Probability images skipped.")
    flatten_into_folder([d.load_labels(minimum_blob_overlap) for d in all_data], flat_labels)


def run_ep_evaluation(result_name, result_folder_prefix, output_path, draw_details=True):
    input_file_part = "tif"
    algo_result_part = "tif"
    algo_name = result_name
    algo_dir = result_folder_prefix + "_labels"

    gt_parameters = "/Parser LABEL \"data_annotation\" \"{0}\" NONE".format(input_file_part)
    algo_parameters = "{0} /Parser LABEL \"{1}\" \"{2}\" NONE".format(algo_name, algo_dir, algo_result_part)
    input_parameters = "/Input data_input \"{0}\"".format(input_file_part)
    if not draw_details:
        input_parameters = ""

    parameters = "{0} {1} {2} {3}".format(output_path, gt_parameters, algo_parameters, input_parameters)

    ep_evaluate_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "vendor", "ep", "evaluate.py")
    final_call = ep_evaluate_path + " " + parameters
    print(final_call)

    subprocess.run('python3 ' + final_call, stderr=subprocess.STDOUT, shell=True,
                   check=True, bufsize=0)


def discard_evaluation_images(result_folder_prefix, output_path, discard_probs, discard_labels):
    prob_dir = os.path.join(output_path, result_folder_prefix + FLATTEN_PROB_SUFIX)
    labels_dir = os.path.join(output_path, result_folder_prefix + FLATTEN_LABELS_SUFIX)

    if discard_probs and os.path.isdir(prob_dir):
        prob_files = [os.path.join(prob_dir, f) for f in os.listdir(prob_dir) if f.endswith(".tif")]
        if prob_files:
            print("Discarding produced prob files: ", len(prob_files))
            shutil.rmtree(prob_dir)

    if discard_labels:
        labels_files = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith(".tif")]
        if labels_files:
            # In case of labels only remove files as evalution summary is there too.
            print("Discarding produced labels files: ", len(labels_files))
            for labels_file in labels_files:
                os.remove(labels_file)


def parse_arguments(provided_args=None):
    parser = argparse.ArgumentParser(description='CLB evaluation of the probability and labels.', add_help=True)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--data',
                          help='path to directory with train, val and test dirs, '
                               'required if --annotated_input or --annotated_gt not specified')
    required.add_argument('--datasets',
                          help='used and required if --data is specified, describes which datasets should be used:'
                               'e.g. "train+val", "test", "val+test"')
    required.add_argument('--annotated_input', help='path to folder with input imagery, ignored if --data provided')
    required.add_argument('--annotated_gt', help='path to folder with annotations, ignored if --data provided')
    required.add_argument('--labels', help='path to folder or volumetric file with final cell labels', required=True)
    required.add_argument('--output', help='folder where all the data and evaluation will be moved', required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--skip_details', dest='skip_details', action='store_true',
                          help='prepare segmentation details')
    optional.add_argument('--discard_probs', dest='discard_probs', action='store_true',
                          help='remove all generated probability files after evaluation')
    optional.add_argument('--discard_labels', dest='discard_labels', action='store_true',
                          help='remove all generated label files after evaluation, evaluation results are still preserved')
    optional.add_argument('--minimum_blob_overlap',
                          type=float,
                          help='path to folder with prediction for input imagery',
                          default=0.6)
    optional.add_argument('--probs', help='path to folder with prediction for input imagery')
    optional.add_argument('--identify_yaml', help='path to yaml file with identify_cells parameters')
    optional.add_argument('--regen_prob', dest='regen_prob', action='store_true',
                          help="rerun prediction using provided params")
    optional.add_argument('--regen_labels', dest='regen_labels', action='store_true',
                          help="rerun postprocessing using provided params", default=False)
    required.add_argument('--model',
                          help='h5 model of the network to use for prediction (expects also the existence of corresponding yaml files)')
    required.add_argument('--name', help='friendly name of the labels provider', default="algo")
    parser.set_defaults(regen_prob=False)
    parser.set_defaults(regen_labels=False)
    parser.set_defaults(skip_details=False)
    parser.set_defaults(discard_probs=False)
    parser.set_defaults(discard_labels=False)
    parsed_args = parser.parse_args(provided_args)
    parsed_args.regen_labels |= parsed_args.regen_prob

    if parsed_args.data is None and (parsed_args.annotated_input is None or parsed_args.annotated_gt is None):
        raise Exception("Either '--data' or both '--annotated_input' and '--annotated_gt' has to be provided.")

    return parsed_args


def main(args):
    probs_temp = None
    try:
        # If folder for probabilities not provided use temporary one.
        if args.probs is None:
            probs_temp = tempfile.mkdtemp()
            args.probs = probs_temp

        if args.data is not None:
            datasets_list = (
                    args.datasets or "T8/train+T8/val+T8/test+T3/train+T3/val+T5/train+T5/val+T6/train+T6/val").split(
                "+")
            all_data = []
            for dataset in datasets_list:
                dataset_path = os.path.join(args.data, dataset)
                all_data += get_all_datasets(os.path.join(dataset_path, "images"),
                                             os.path.join(dataset_path, "labels"),
                                             args.probs, args.labels)
        else:
            all_data = get_all_datasets(args.annotated_input, args.annotated_gt, args.probs, args.labels)

        print("Loaded data:")
        for d in all_data:
            print("    Data name: " + os.path.basename(d.input))
            print("        " + str(d))

        ensure_probabilites(all_data, args)
        ensure_labels(all_data, args)

        flatten_all_data(all_data, args.minimum_blob_overlap, args.name, args.output)
        run_ep_evaluation(args.name, args.name, args.output, draw_details=not args.skip_details)

        discard_evaluation_images(args.name, args.output,
                                  discard_probs=args.discard_probs,
                                  discard_labels=args.discard_labels)
    finally:
        keras.backend.clear_session()
        load_model_with_cache.cache_clear()
        if probs_temp is not None:
            shutil.rmtree(probs_temp)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

