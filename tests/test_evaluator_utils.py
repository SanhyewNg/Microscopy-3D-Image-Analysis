import pytest
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from clb.evaluate.evaluator_utils import EpEvalFormatter, EpEvaluation
from clb.classify.utils import ClassVolumeSet


@pytest.mark.evaluator
class TestEpEvalFormatter(unittest.TestCase):
    def setUp(self):
        self.test_data = Path(__file__).resolve().parents[0] / "data" / "evaluator"
        self.tmp_folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_folder)

    def test_formatter_from_class_volume_sets(self):
        cvs = ClassVolumeSet(
            input="dummy/path/img.tif",
            crop_info=None,
            gt=str(self.test_data / "gt_2.tif"),
            label=str(self.test_data / "segmented_2.tif"),
            classes=str(self.test_data / "classified_2.tif"),
        )
        cvs.cell_positions = {
            1: {"id": 1, "pos_x": 24.5, "pos_y": 24.5, "pos_z": 1.0},
            2: {"id": 2, "pos_x": 24.5, "pos_y": 74.5, "pos_z": 1.0},
            4: {"id": 4, "pos_x": 24.5, "pos_y": 74.5, "pos_z": 0.0},
        }
        cvs.cell_classes = {
            1: {"id": 1, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
            2: {"id": 2, "class": 0, "class_fraction": 0, "class_pixels": 0},
        }
        cvs.cell_classes_predicted = {
            1: {"id": 1, "class": 0, "class_fraction": 0, "class_pixels": 0},
            2: {"id": 2, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
        }

        EpEvalFormatter().from_class_volume_sets(class_volume_sets=[cvs], output_path=self.tmp_folder)
        gt_file = Path(self.tmp_folder, "gt.tsv")
        pred_file = Path(self.tmp_folder, "pred.tsv")
        self.assertTrue(gt_file.exists())
        self.assertTrue(pred_file.exists())
        with open(str(gt_file), "r") as gt_f:
            self.assertEqual("Frame_number;Cell_number;Position_X;Position_Y\nimg;1;24.5;24.5\n", gt_f.read())
        with open(str(pred_file), "r") as pred_f:
            self.assertEqual("Frame_number;Cell_number;Position_X;Position_Y\nimg;2;24.5;74.5\n", pred_f.read())

    def test_formatter_from_csv(self):
        EpEvalFormatter().from_csv(
            csv_file_path_with_predictions=str(self.test_data / "input.tsv"), output_path=str(self.tmp_folder)
        )
        gt_file = Path(self.tmp_folder, "gt.tsv")
        pred_file = Path(self.tmp_folder, "pred.tsv")
        try:
            gt_f = open(str(gt_file), "r")
            expected_gt = open(str(self.test_data / "gt.tsv"), "r")
            pred_f = open(str(pred_file), "r")
            expected_pred = open(str(self.test_data / "pred.tsv"), "r")
            self.assertEqual(gt_f.read(), expected_gt.read())
            self.assertEqual(pred_f.read(), expected_pred.read())

        finally:
            gt_f.close()
            expected_gt.close()
            pred_f.close()
            expected_pred.close()


@pytest.mark.evaluator
class TestEpEvaluation(unittest.TestCase):
    @mock.patch("subprocess.run", return_value=None)
    def test_run_default(self, patched_system):
        EpEvaluation().run("a/b/c", "result_foldername")
        command = patched_system.call_args[0][0].split("evaluate.py")[1].strip()
        self.assertEqual(
            command,
            'a/b/c "result_foldername" "gt" NONE result_foldername "result_foldername" "pred" NONE',
        )

    @mock.patch("subprocess.run", return_value=None)
    def test_run_custom_names(self, patched_system):
        EpEvaluation("groundtruth.tsv", "predicted.tsv").run("a/b/c", "result_foldername")
        command = patched_system.call_args[0][0].split("evaluate.py")[1].strip()
        self.assertEqual(
            command,
            'a/b/c "result_foldername" "groundtruth" NONE result_foldername "result_foldername" "predicted" NONE',
        )

    @mock.patch("subprocess.run", return_value=None)
    def test_run_with_input_data(self, patched_system):
        EpEvaluation(input_data_available=True).run("a/b/c", "result_foldername")
        command = patched_system.call_args[0][0].split("evaluate.py")[1].strip()
        self.assertEqual(
            command,
            'a/b/c "result_foldername" "gt" NONE result_foldername "result_foldername" "pred" NONE /Input data_input "tif"',
        )