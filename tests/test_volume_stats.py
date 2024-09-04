import pytest
import unittest

import numpy as np

import clb.stats.volume_stats as vs


@pytest.mark.statistics
class TestCalculateCellsVolume(unittest.TestCase):
    def setUp(self):
        self.empty_volume = np.zeros((3, 3, 3))
        self.volume = np.array([
            [
                [0, 0, 1, 1],
                [2, 0, 1, 1],
                [2, 0, 0, 0],
                [0, 0, 0, 3]
            ],
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 4],
                [0, 0, 0, 0]
            ]
        ])
        self.ones_volume = np.ones((2, 2, 2))

    def test_empty_volume(self):
        cells_num = vs.get_num_of_cell_pixels(self.empty_volume)
        expected_cells_num = 0

        self.assertEqual(cells_num, expected_cells_num)

    def test_non_empty_volume(self):
        cells_num = vs.get_num_of_cell_pixels(self.volume)
        expected_cells_num = 12

        self.assertEqual(cells_num, expected_cells_num)

    def test_ones_volume(self):
        cells_num = vs.get_num_of_cell_pixels(self.ones_volume)
        expected_cells_num = 8

        self.assertEqual(cells_num, expected_cells_num)

    def test_non_empty_volume_with_ids(self):
        cells_num = vs.get_num_of_cell_pixels(self.volume, ids={2, 3})
        expected_cells_num = 3

        self.assertEqual(cells_num, expected_cells_num)
