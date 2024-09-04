import os
import pytest
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

import clb.stats.spatial_stats as spat_stats


@pytest.mark.statistics
class TestCalculateEsd(unittest.TestCase):
    def test_without_pixel_size(self):
        class_mask = np.array([
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 0]
        ])

        output = spat_stats.calculate_esd(class_mask)
        expected = np.array([
            [1.4142, 1, 1.4142],
            [1, 0, 1],
            [1.4142, 1, 1.4142]
        ])

        np.testing.assert_almost_equal(output, expected, decimal=3)

    def test_with_equal_pixel_sizes(self):
        class_mask = np.array([
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 0]
        ])
        pixel_sizes = (0.5, 0.5)

        output = spat_stats.calculate_esd(class_mask, pixel_sizes)
        expected = np.array([
            [0.7071, 0.5, 0.7071],
            [0.5, 0, 0.5],
            [0.7071, 0.5, 0.7071]
        ])

        np.testing.assert_almost_equal(output, expected, decimal=3)

    def test_with_different_pixel_sizes(self):
        class_mask = np.array([
            [0, 0, 0],
            [0, 0, 255],
            [0, 0, 0]
        ])
        pixel_sizes = (0.5, 1)

        output = spat_stats.calculate_esd(class_mask, pixel_sizes)
        expected = np.array([
            [2.0615, 1.1180, 0.5],
            [2, 1, 0],
            [2.0615, 1.1180, 0.5]
        ])

        np.testing.assert_almost_equal(output, expected, decimal=3)

    def test_3d_input_without_pixel_size(self):
        class_mask = np.array([
            [
                [0, 0, 0],
                [0, 255, 0],
                [0, 0, 0]
            ],
            [
                [255, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        ])

        output = spat_stats.calculate_esd(class_mask)
        expected = np.array([
            [
                [1, 1, 1.4142],
                [1, 0, 1],
                [1.4142, 1, 1.4142]
            ],
            [
                [0, 1, 1.73205],
                [1, 1, 1.4142],
                [1.73205, 1.4142, 1.73205]
            ]
        ])

        np.testing.assert_almost_equal(output, expected, decimal=3)


@pytest.mark.statistics
class TestCalculateClassDistances(unittest.TestCase):
    def test_2d_input(self):
        class_ids = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 2],
            [3, 0, 0, 2]
        ])
        distances = np.array([
            [0.2, 0.5, 1, 1],
            [1, 0.19, 0.5, 0.2],
            [1, 0, 2, 3],
            [0.3, 0.4, 0.5, 0.6]
        ])

        output = spat_stats.calculate_class_distances(class_ids, distances)
        expected = [0.19, 0.3, 0.6]

        np.testing.assert_equal(np.sort(output), np.sort(expected))

    def test_3d_input(self):
        class_ids = np.array([
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [2, 2, 2, 0],
                [0, 0, 0, 3]
            ],
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 2, 4],
                [0, 0, 0, 3]
            ]
        ])
        distances = np.array([
            [
                [0.1, 2, 1, 3],
                [0.5, 0.3, 0.2, 0.3],
                [2.2, 2.3, 2.4, 3],
                [0, 3, 2.5, 0.7]
            ],
            [
                [1.2, 2.3, 3.4, 4.5],
                [0.2, 0.3, 0.4, 0.5],
                [0.7, 2, 2.2, 4.4],
                [0.2, 0.1, 0.3, 3.3]
            ]
        ])

        output = spat_stats.calculate_class_distances(class_ids, distances)
        expected = [0.1, 2.2, 0.7, 4.4]

        np.testing.assert_equal(np.sort(output), np.sort(expected))


@pytest.mark.statistics
class TestGetClassLabels(unittest.TestCase):
    def test_no_class_ids(self):
        cell_ids = np.array([
            [0, 0, 1, 1],
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 3, 3]
        ])
        class_mask = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        class_ids = spat_stats.get_class_labels(cell_ids, class_mask)
        expected = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        np.testing.assert_equal(class_ids, expected)

    def test_with_class_ids(self):
        cell_ids = np.array([
            [0, 0, 1, 1],
            [2, 2, 1, 1],
            [0, 2, 0, 0],
            [0, 3, 3, 3]
        ])
        class_mask = np.array([
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 0, 0],
            [0, 255, 255, 255]
        ])

        class_ids = spat_stats.get_class_labels(cell_ids, class_mask)
        expected = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 3, 3, 3]
        ])

        np.testing.assert_equal(class_ids, expected)


@pytest.mark.statistics
class TestRemoveDoublePositives(unittest.TestCase):
    @staticmethod
    def test_remove_double_positives():
        class_mask = np.array([
            [255, 255, 0],
            [255, 255, 0],
            [0, 0, 2]
        ])
        ref_class_mask = np.array([
            [2, 2, 0],
            [2, 2, 0],
            [0, 0, 255]
        ])

        result = spat_stats.remove_double_positives(class_mask, ref_class_mask)
        expected = np.array([
            [255, 255, 0],
            [255, 255, 0],
            [0, 0, 2]
        ])

        np.testing.assert_equal(result, expected)

    @staticmethod
    def test_with_duplicates():
        class_mask = np.array([
            [255, 255, 0, 0],
            [0, 255, 0, 0],
            [0, 0, 255, 255],
            [2, 2, 2, 0]
        ])
        ref_class_mask = np.array([
            [255, 255, 0, 0],
            [0, 255, 0, 0],
            [0, 0, 2, 2],
            [255, 255, 255, 0]
        ])

        result = spat_stats.remove_double_positives(class_mask, ref_class_mask)
        expected = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 255, 255],
            [2, 2, 2, 0]
        ])

        np.testing.assert_equal(result, expected)


@pytest.mark.statistics
class TestWholeScript(unittest.TestCase):
    def setUp(self):
        self.results_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results_dir)

    def test(self):
        graph_path = os.path.join(self.results_dir, 'graph.png')
        data_path = os.path.join(self.results_dir, 'distances')
        tested_classes = {
            'blue': 'tests/test_statistics/spatial/blue.tif',
            'magenta': 'tests/test_statistics/spatial/magenta.tif',
            'green': 'tests/test_statistics/spatial/green.tif'
        }

        spat_stats.main(
            input='tests/test_statistics/spatial/cells.tif',
            output_graph=graph_path,
            output_data_dir=data_path,
            series=0,
            labels='tests/test_statistics/spatial/labels.tif',
            ref_plot_name='base',
            ref_class_path='tests/test_statistics/spatial/red.tif',
            filter_double_positives=True,
            sort_legend=True,
            tested_classes_names=tested_classes.keys(),
            tested_classes_paths=tested_classes.values()
        )

        result = cv2.imread(graph_path)
        expected = cv2.imread('tests/test_statistics/spatial/graph.png')
        np.testing.assert_equal(result, expected)

        expected_data_dir = 'tests/test_statistics/spatial/distances'
        for result_path, expected_path in zip(sorted(os.listdir(data_path)),
                                              sorted(os.listdir(expected_data_dir))):
            result = pd.read_csv(os.path.join(self.results_dir, 'distances',
                                              result_path))
            expected = pd.read_csv(os.path.join(expected_data_dir, expected_path))

            pd.testing.assert_frame_equal(result, expected)
