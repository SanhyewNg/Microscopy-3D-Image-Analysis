import logging
import os
import shutil
import subprocess
from pathlib import Path

import daiquiri
import imageio
import pandas as pd

from clb.utils import bbox

logger = daiquiri.getLogger(__name__)


class EpEvalFormatter:
    """Evaluator Formatter produce files that are consumed by external evaluator."""

    def __init__(self, output_gt_filename="gt.tsv", output_predicted_filename="pred.tsv", append_index=False):
        self.out_gt_filename = output_gt_filename
        self.out_pred_filename = output_predicted_filename
        self.append_index = append_index

    def from_csv(self, csv_file_path_with_predictions, output_path):
        """Export tsv files that can be consumed by Evaluation Platform.
        
        Args:
            csv_file_path_with_predictions (str): Path to CSV file. First two columns denote multi-index. 
                                                  Columns with label `class` and `pred` shall be present.
            output_path (str): Path to folder where result TSV files will be stored.
        """
        logger.info("Formatting predictions CSV file for evaluator...")
        data = pd.read_csv(csv_file_path_with_predictions, sep="\t", index_col=[0, 1])
        gt_data = data[data["class"] == 1].copy()
        pred_data = data[data["pred"] == 1].copy()
        ground_truth = self._create_output_frame(gt_data)
        predicted = self._create_output_frame(pred_data)
        self._export(ground_truth, output_path, self.out_gt_filename)
        self._export(predicted, output_path, self.out_pred_filename)

    def from_class_volume_sets(self, class_volume_sets, output_path):
        """Export tsv files that can be consumed by Evaluation Platform."""
        logger.info("Formatting ClassVolumeSets for evaluator...")
        ground_truth = self._prepare_export_frame(class_volume_sets, "cell_classes")
        predicted = self._prepare_export_frame(class_volume_sets, "cell_classes_predicted")
        self._export(ground_truth, output_path, self.out_gt_filename)
        self._export(predicted, output_path, self.out_pred_filename)

    def _filter_non_target_classes(self, df):
        return df[df["class"] == 1]

    def _prepare_export_frame(self, class_volume_sets, cell_attrib):
        data = pd.DataFrame()
        for index, cvs in enumerate(class_volume_sets):
            cell_frame = self._create_frame_from_class_volume_set(cvs, cell_attrib, index)
            target_cells = self._filter_non_target_classes(cell_frame)
            with_position = self._join_cell_position(target_cells, cvs)
            data = pd.concat([data, with_position])
        return self._create_output_frame(data)

    def _export(self, export_frame, output_path, filename):
        os.makedirs(output_path, exist_ok=True)
        export_frame.to_csv(os.path.join(output_path, filename), index=False, sep=";")
        logger.info("Exported frame: {}".format(output_path))

    def _format_filenames(self, data):
        filenames = data.index.to_frame()
        try:
            filenames = filenames["filename"]
        except KeyError:
            filenames = filenames[0]
        return (
            filenames.str.replace(".tif", "")
            .str.replace(" 0.5um_dapi_stack", "")
            .str.replace("1024 ", "")
            .str.replace("2048", "")
        )

    def _create_output_frame(self, data):
        export_frame = pd.DataFrame()
        export_frame["Frame_number"] = self._format_filenames(data)
        export_frame["Cell_number"] = data.index.get_level_values(1)  # cell ID
        export_frame["Position_X"] = data["pos_x"]
        export_frame["Position_Y"] = data["pos_y"]
        return export_frame

    def _join_cell_position(self, df, cvs):
        cell_position_frame = pd.DataFrame.from_dict(cvs.cell_positions).T
        cell_position_frame.set_index("id", inplace=True)
        return df.join(cell_position_frame, on="id")

    def _create_frame_from_class_volume_set(self, class_volume_set, cell_attrib, index):
        df = pd.DataFrame(getattr(class_volume_set, cell_attrib)).T
        df["filename"] = Path(class_volume_set.input).name
        if self.append_index:
            df["filename"] += '_' + str(index)
        df["id"] = df.id.astype(int)
        df.set_index(["filename", "id"], inplace=True)
        return df


class EpEvaluation:
    def __init__(self, gt_filename="gt.tsv", predicted_filename="pred.tsv", input_data_available=False):
        self.gt_filename_stem = Path(gt_filename).stem
        self.pred_filename_stem = Path(predicted_filename).stem
        self.input_data_folder = "data_input" if input_data_available else None

    def run(self, output_path, result_folder_name):
        ep_evaluate_path = self._get_evaluator_path()
        gt_params, algo_params, input_parameters = self._get_eval_call_params(result_folder_name)

        ep_eval_call = "{ep_evaluate_path} {output_path} {gt_params} {algo_params} {input_parameters}".format(
            ep_evaluate_path=ep_evaluate_path, output_path=output_path, gt_params=gt_params, algo_params=algo_params,
            input_parameters=input_parameters
        )
        logger.info("Calling ep evaluator: {}".format(ep_eval_call))
        subprocess.run(
            "python3 " + ep_eval_call, stderr=subprocess.STDOUT, shell=True, check=True, bufsize=0, timeout=60
        )

    def _get_evaluator_path(self):
        return str(Path(__file__).resolve().parents[2] / "vendor" / "ep" / "evaluate.py")

    def _get_eval_call_params(self, result_folder_name):
        algo_name = result_folder_name
        algo_dir = result_folder_name

        input_parameters = "" if self.input_data_folder is None \
            else '/Input {0} "{1}"'.format(self.input_data_folder, 'tif')
        gt_params = '"{0}" "{1}" NONE'.format(algo_dir, self.gt_filename_stem)
        pred_params = '{0} "{1}" "{2}" NONE'.format(algo_name, algo_dir, self.pred_filename_stem)
        return gt_params, pred_params, input_parameters


class CopyClassVolumeSetFiles:
    def __init__(self, copy_methods):
        self.copy_methods = copy_methods

    def __call__(self, class_volume_set, index):
        for copy in self.copy_methods:
            copy(class_volume_set, index)


class CopyAnnotatedImageFromClassVolumeSet:
    def __init__(self, target_folder, attrib, single_slice=False):
        self.target_folder = target_folder
        self.attrib = attrib
        self.single_slice = single_slice
        os.makedirs(target_folder, exist_ok=True)

    def __call__(self, class_volume_set, index):
        image_file = getattr(class_volume_set, self.attrib)
        annotation_image = getattr(class_volume_set, "gt")
        self.copy_annotated_slices(image_file, annotation_image, index)

    def copy_annotated_slices(self, image_file, annotation_image, index):
        image = imageio.volread(image_file)
        annotation = imageio.volread(annotation_image)
        annotated_slices = self._get_annotated_slices(annotation)
        annotated = image[annotated_slices, ...]
        annotated_output_path = self._get_image_output_path(image_file, index)
        imageio.mimwrite(annotated_output_path, annotated)

    def _get_image_output_path(self, image_file, index):
        path = Path(image_file)
        name = "{}_{}{}".format(path.stem, index, path.suffix)
        return os.path.join(self.target_folder, name)

    def _get_annotated_slices(self, image):
        annotated = bbox(image, 0)
        if annotated is not None:
            first_annotated_z, last_annotated_z = annotated
            if self.single_slice:
                first_annotated_z = last_annotated_z = (last_annotated_z + first_annotated_z) // 2
            return tuple(range(first_annotated_z, last_annotated_z + 1))
        else:
            logger.warn("No annotated slice found")


def copy_annotated_input_and_annotations_to_output(output, class_volume_sets, copy_single_slice):
    copy = CopyClassVolumeSetFiles(
        copy_methods=(
            CopyAnnotatedImageFromClassVolumeSet(os.path.join(output, "data_input"), "input", copy_single_slice),
            CopyAnnotatedImageFromClassVolumeSet(os.path.join(output, "data_annotation"), "gt", copy_single_slice),
        )
    )
    for index, cvs in enumerate(class_volume_sets):
        copy(cvs, index)


def remove_segmentation_folder(labels_path):
    logger.info("Removing segmentation folder: {}".format(labels_path))
    shutil.rmtree(labels_path)
