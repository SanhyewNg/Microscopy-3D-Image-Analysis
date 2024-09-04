import argparse
import logging
import os
import shutil
from pathlib import Path
from pprint import pformat

import daiquiri
import imageio
import pandas as pd
from tqdm import tqdm

from clb.classify.classify import get_parser
from clb.classify.classify import main as classifier
from clb.classify.extractors import DESIRED_VOXEL_UM
from clb.classify.feature_extractor import extract_shape_features
from clb.classify.prepare_train import determine_cell_classes, extract_cell_predicted_classes
from clb.classify.utils import ClassVolumeSet, get_all_multiple_datasets
from clb.evaluate.evaluator_utils import (
    EpEvalFormatter,
    EpEvaluation,
    copy_annotated_input_and_annotations_to_output,
    remove_segmentation_folder,
)

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


def parse_arguments(provided_args=None):
    parser = argparse.ArgumentParser(description="Evaluation for classification.", add_help=True)
    required = parser.add_argument_group("required arguments")
    required.add_argument("--name", help="friendly name of the classificator", default="classificator")
    required.add_argument("--data", help="path to root data directory which contain dataset folders")
    required.add_argument(
        "--datasets",
        help="used and required if --data is specified. "
        'Describes which datasets from root data dir should be used: e.g. "train+test", "test", "train", "NC1/train+test"',
    )
    required.add_argument("--labels", help="path to folder with final cell labels")
    required.add_argument("--output", help="folder where all the data and evaluation will be moved", required=True)
    required.add_argument(
        "--model",
        help="path to classifier model in form of pkl to use for prediction"
        " (expects also the existence of corresponding yaml files)",
    )
    required.add_argument("--class_name", help="classifier class name e.g.: cd8, pdl1")
    required.add_argument(
        "--overlap_treshold",
        help="Treshold over which cell is assigned as class. Default = 0.5",
        default=0.5,
        type=float,
    )
    required.add_argument(
        "--discard_labels",
        dest="discard_labels",
        action="store_true",
        help="remove all generated label files after evaluation, evaluation results are still preserved",
    )

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("--classes", help="path to folder with cell classes")

    parser.set_defaults(discard_labels=False)
    parsed_args = parser.parse_args(provided_args)
    return parsed_args


class DataLoader:
    def __init__(self, data=None, datasets=None, labels=None, classes=None):
        """
        Load data. If `data` or `dataset` not given, deduce path.

        Args: 
            data (str): Path to root data directory
            datasets (str): Relative path from root dir to datasets
            labels (str): Path to folder with segmentation images (labels)
            classes (str): Path to folder with classification images 
        """
        self.data = data if data is not None else self._get_data()
        self.datasets = datasets.split("+") if datasets is not None else self._get_datasets()
        self.labels = labels if labels else os.path.join(self.data, "labels")
        self.classes = classes if classes else os.path.join(self.data, "classes")

    def prepare(self, class_name, input_folder="input", ground_truth_folder="annotations"):
        """
        Prepare dataset

        Args:
            class_name (str): Cell class name based on marker in image
            input_folder (str): Folder name with input images
            ground_truth_folder (str): Folder name with ground truth images

        Return:
            List of ClassVolumeSet
        """
        class_volume_sets = get_all_multiple_datasets(
            root_path=self.data,
            names=self.datasets,
            input_folder_sufix=input_folder,
            ground_truth_folder_sufix=ground_truth_folder,
            labels=self.labels,
            classes=self.classes,
            class_name=class_name,
        )

        if not len(class_volume_sets):
            logger.error("No ClassVolumeSets found !")
            raise ValueError

        cvs_names = [os.path.basename(cvs.input) for cvs in class_volume_sets]
        logger.info("Dataset prepared: \n{}".format(pformat(cvs_names)))
        logger.debug(pformat(list(map(str, class_volume_sets))))

        return class_volume_sets

    def _get_data(self):
        """Gets default root evaluation path for classification."""
        root_data = str(Path(__file__).resolve().parents[2] / "data" / "classification" / "evaluation")
        logger.info("Used automatic discovery of root data folder: {}".format(root_data))
        return root_data

    def _get_datasets(self, data_folders=("train", "test")):
        """Create list of relative path to datasets from root data directory."""
        data_root = Path(self.data)
        datasets_subdirs = [
            str(abs_path.relative_to(data_root)) for f in data_folders for abs_path in list(data_root.rglob(f))
        ]
        logger.info("Used automatic discovery of datasets subdirs: {}".format(datasets_subdirs))
        return datasets_subdirs


class ValidateClassesAndLabels:
    """Ensure that classes and labels exists for given class_validation_sets"""

    def __init__(self, classification_model):
        """ 
        Args:
           classification_model (str): Path to classification model e.g.: /a/b/model.pkl
        """
        self.classification_model = classification_model
        self.classifier_parser = get_parser()

    def __call__(self, class_validation_sets):
        for cvs in tqdm(class_validation_sets, "Calculating classification"):
            if not self._segmentation_image_exist(cvs) or not self._classified_image_exist(cvs):
                logger.debug("No classes or labels found for: {}, generating ...".format(cvs.input))
                args = self.classifier_parser.parse_args(
                    [
                        "--input",
                        cvs.input,
                        "--outputs",
                        cvs.classes,
                        "--model",
                        self.classification_model,
                        "--labels",
                        cvs.label,
                    ]
                )
                classifier(args)

    def _classified_image_exist(self, cvs):
        return os.path.isfile(cvs.classes)

    def _segmentation_image_exist(self, cvs):
        return os.path.isfile(cvs.label)


class ValidateCellPosition:
    """Ensure that cell_positions exists for given class_validation_sets"""

    def __call__(self, class_validation_sets):
        for cvs in tqdm(class_validation_sets, "Calculating cell positions"):
            if not cvs.cell_positions:
                logger.debug("No cell_positions found for given Class Volume Set: {}".format(cvs.input))
                shape_features = extract_shape_features(labels_volume=imageio.volread(cvs.label), features_type="")
                for k, v in shape_features.items():
                    cvs.cell_positions[k] = {
                        "id": v["id"],
                        "pos_x": v["pos_x"],
                        "pos_y": v["pos_y"],
                        "pos_z": v["pos_z"],
                    }


class EvaluatorClassify:
    def __init__(self, class_volume_sets, validators=None, overlap_treshold=0.5):
        """
        Args:
            class_volume_sets (list): List of ClassVolumeSet.
            validators (list): List of callables that validates ClassVolumeSet. 
        """
        self.class_volume_sets = class_volume_sets
        self.validators = validators
        self.is_data_validated = False
        self.overlap_treshold = overlap_treshold

    def evaluate(self, output_path):
        if not self.is_data_validated:
            logger.warn("Class Volume Sets is not validated. Please Validate it first")
            exit(321)
        self._get_cell_classes_statistics()
        self._filter_cell_classes_predicted()
        EpEvalFormatter(append_index=True).from_class_volume_sets(class_volume_sets=self.class_volume_sets, output_path=output_path)
        EpEvaluation(input_data_available=True).run(output_path=str(Path(output_path).parent), result_folder_name=str(Path(output_path).name))

    def validate_input(self):
        if len(self.class_volume_sets) == 0:
            logger.warn("Validators not applied. Class Volume Sets is empty")
            exit(871)
        for validator in self.validators:
            logger.info("Appling validator {}".format(str(validator)))
            validator(self.class_volume_sets)
        self.is_data_validated = True

    def _get_cell_classes_statistics(self):
        """Gets cell_classes and cell_classes_predicted"""
        determine_cell_classes(self.class_volume_sets, overlap_threshold=self.overlap_treshold)
        extract_cell_predicted_classes(self.class_volume_sets)

    def _filter_cell_classes_predicted(self):
        """Filter out predicted cells that are not in gt (cell_classes)"""
        for cvs in self.class_volume_sets:
            to_remove = set(cvs.cell_classes_predicted.keys()).difference(set(cvs.cell_classes.keys()))
            for rm in to_remove:
                cvs.cell_classes_predicted.pop(rm, None)


def main(
    model,
    output,
    class_name,
    name,
    data=None,
    datasets=None,
    labels=None,
    classes=None,
    overlap_treshold=0.5,
    discard_labels=False,
):
    """Run evaluator classify. Copy `gt` and `input` to output folder.
    
    Args: please refer to parse_arguments function docs
    """
    logger.info("Evaluator classify started for model: {}".format(model))

    eval_output = os.path.join(output, name) if name is not None else output
    labels_folder = labels if labels is not None else os.path.join(output, "segmented")
    classes_folder = classes if classes is not None else eval_output
    data_loader = DataLoader(data=data, datasets=datasets, labels=labels_folder, classes=classes_folder)
    class_volume_sets = data_loader.prepare(class_name=class_name)

    copy_annotated_input_and_annotations_to_output(output, class_volume_sets, copy_single_slice=True)

    validators = (ValidateClassesAndLabels(model), ValidateCellPosition())
    evaluator = EvaluatorClassify(
        class_volume_sets=class_volume_sets, validators=validators, overlap_treshold=overlap_treshold
    )
    evaluator.validate_input()
    evaluator.evaluate(eval_output)

    if discard_labels:
        if labels:
            logger.warn("Attempt to remove precomputed segmentation folder. Abort discard_labels")
        else:
            remove_segmentation_folder(labels_folder)


if __name__ == "__main__":
    args = parse_arguments()
    kwargs = dict(args._get_kwargs())
    main(**kwargs)
