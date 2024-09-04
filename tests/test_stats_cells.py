import pytest
import unittest

import numpy as np
import numpy.testing as nptest

import clb.stats.cells
from clb.utils import replace_values


@pytest.mark.statistics
class TestStatsCells(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

    def test_general_stats(self):
        seg_volume = np.zeros((3, 50, 50), dtype=np.uint16)
        seg_volume[0:2, 10:20, 10:20] = 1
        seg_volume[1, 10:20, 30:50] = 3

        class_volume = seg_volume.copy().astype(np.float32)
        class_volume = replace_values(class_volume, {1: 0.6, 3: 0.3})

        nptest.assert_array_equal(len(np.unique(seg_volume)), len(np.unique(class_volume)))

        stats = clb.stats.cells.cell_stats(seg_volume, class_volume)
        self.assertEqual(2, stats['cell_number'])
        self.assertEqual(1, stats['class_cell_number'])

        # for slice 0
        self.assertEqual(1, stats['slices'][0]['cell_number'])
        self.assertEqual(1, stats['slices'][0]['class_cell_number'])

        # for slice 1
        self.assertEqual(2, stats['slices'][1]['cell_number'])
        self.assertEqual(1, stats['slices'][1]['class_cell_number'])

        # for slice 2
        self.assertEqual(0, stats['slices'][2]['cell_number'])
        self.assertEqual(0, stats['slices'][2]['class_cell_number'])

    def tearDown(self):
        pass
