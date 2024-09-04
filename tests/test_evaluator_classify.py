import unittest
from pathlib import Path
import pytest
from unittest import mock

from clb.evaluate.evaluator_classify import EvaluatorClassify
from clb.classify.utils import ClassVolumeSet


@pytest.mark.classification
@pytest.mark.evaluator
class TestEvaluatorClassify(unittest.TestCase):
    def setUp(self):
        self.test_data = Path(__file__).resolve().parents[0] / "data" / "evaluator"

    def test__get_cell_classes_statistics_all_cells(self):
        cvs = ClassVolumeSet(
            input="",
            crop_info=None,
            gt=str(self.test_data / "gt_1.tif"),
            label=str(self.test_data / "segmented_1.tif"),
            classes=str(self.test_data / "classified_1.tif"),
        )
        evaluator = EvaluatorClassify(class_volume_sets=[cvs], validators=None, overlap_treshold=0.5)
        evaluator._get_cell_classes_statistics()
        cell_metrics = evaluator.class_volume_sets[0]

        self.assertDictEqual(
            cell_metrics.cell_classes,
            {
                1: {"id": 1, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
                2: {"id": 2, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
                3: {"id": 3, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
                4: {"id": 4, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
            },
        )
        self.assertDictEqual(
            cell_metrics.cell_classes_predicted,
            {
                1: {"id": 1, "class": 0, "class_fraction": 0, "class_pixels": 0},
                2: {"id": 2, "class": 0, "class_fraction": 0, "class_pixels": 0},
                3: {"id": 3, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
                4: {"id": 4, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
            },
        )

    def test__get_cell_classes_statistics_missing_cells(self):
        cvs = ClassVolumeSet(
            input="",
            crop_info=None,
            gt=str(self.test_data / "gt_2.tif"),
            label=str(self.test_data / "segmented_2.tif"),
            classes=str(self.test_data / "classified_2.tif"),
        )

        evaluator = EvaluatorClassify(class_volume_sets=[cvs], validators=None, overlap_treshold=0.5)
        evaluator._get_cell_classes_statistics()
        cell_metrics = evaluator.class_volume_sets[0]

        self.assertDictEqual(
            cell_metrics.cell_classes,
            {
                1: {"id": 1, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
                2: {"id": 2, "class": 0, "class_fraction": 0, "class_pixels": 0},
            },
        )
        self.assertDictEqual(
            cell_metrics.cell_classes_predicted,
            {
                1: {"id": 1, "class": 0, "class_fraction": 0, "class_pixels": 0},
                2: {"id": 2, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
            },
        )

    def test__filter_cell_classes_predicted_equal(self):
        cvs = ClassVolumeSet(input="", crop_info=None, gt="", label="", classes="")
        cvs.cell_classes = {1: "cell 1", 2: "cell 2", 3: "cell 3"}
        cvs.cell_classes_predicted = {1: "cell 1", 2: "cell 2", 3: "cell 3"}
        evaluator = EvaluatorClassify(class_volume_sets=[cvs], validators=None, overlap_treshold=0.5)
        evaluator._filter_cell_classes_predicted()
        cell_metrics = evaluator.class_volume_sets[0]

        self.assertEqual(list(cell_metrics.cell_classes.keys()), [1, 2, 3])

    def test__filter_cell_classes_predicted_trim_superset(self):
        cvs = ClassVolumeSet(input="", crop_info=None, gt="", label="", classes="")
        cvs.cell_classes = {1: "cell 1", 2: "cell 2", 3: "cell 3"}
        cvs.cell_classes_predicted = {1: "cell 1"}
        evaluator = EvaluatorClassify(class_volume_sets=[cvs], validators=None, overlap_treshold=0.5)
        evaluator._filter_cell_classes_predicted()
        cell_metrics = evaluator.class_volume_sets[0]

        self.assertEqual(list(cell_metrics.cell_classes.keys()), [1, 2, 3])

    def test__filter_cell_classes_predicted_(self):
        cvs = ClassVolumeSet(input="", crop_info=None, gt="", label="", classes="")
        cvs.cell_classes = {1: "cell 1"}
        cvs.cell_classes_predicted = {1: "cell 1", 2: "cell 2", 3: "cell 3"}
        evaluator = EvaluatorClassify(class_volume_sets=[cvs], validators=None, overlap_treshold=0.5)
        evaluator._filter_cell_classes_predicted()
        cell_metrics = evaluator.class_volume_sets[0]

        self.assertEqual(list(cell_metrics.cell_classes.keys()), [1])

    def test_evaluate_validate(self):
        cvs = ClassVolumeSet(
            input="",
            crop_info=None,
            gt=str(self.test_data / "gt_1.tif"),
            label=str(self.test_data / "segmented_1.tif"),
            classes=str(self.test_data / "classified_1.tif"),
        )
        evaluator = EvaluatorClassify(class_volume_sets=[cvs], validators=None, overlap_treshold=0.5)
        with self.assertRaises(SystemExit) as se:
            evaluator.evaluate("dummy_output_path")
        self.assertEqual(se.exception.code, 321)

    @mock.patch("clb.evaluate.evaluator_classify.EpEvaluation.run", return_value=None)
    @mock.patch("clb.evaluate.evaluator_classify.EpEvalFormatter.from_class_volume_sets", return_value=None)
    def test_evaluate_formatter_call_args(self, patched_formatter, patched_runner):
        cvs = ClassVolumeSet(
            input="dummy/path/img.tif",
            crop_info=None,
            gt=str(self.test_data / "gt_2.tif"),
            label=str(self.test_data / "segmented_2.tif"),
            classes=str(self.test_data / "classified_2.tif"),
        )
        evaluator = EvaluatorClassify(class_volume_sets=[cvs], validators=None, overlap_treshold=0.5)
        evaluator.is_data_validated = True
        evaluator.evaluate("dummy_path")

        self.assertEqual(
            patched_formatter.call_args[1]["class_volume_sets"][0].cell_classes,
            {
                1: {"id": 1, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
                2: {"id": 2, "class": 0, "class_fraction": 0, "class_pixels": 0},
            },
        )
        self.assertEqual(
            patched_formatter.call_args[1]["class_volume_sets"][0].cell_classes_predicted,
            {
                1: {"id": 1, "class": 0, "class_fraction": 0, "class_pixels": 0},
                2: {"id": 2, "class": 1, "class_fraction": 1.0, "class_pixels": 900},
            },
        )