import pytest
import unittest

import numpy as np
import numpy.testing as nptest

from clb.segment.segment_cells import \
    (find_corresponding_labels, make_consistent_labels, label_cells_cc,
     label_cells_watershed, label_cells_by_layers, dilation_only_2d)


@pytest.mark.classification
class TestIdentifyCells(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.image1 = np.zeros((20, 20), dtype=np.uint8)
        self.image1[0:10, 0:10] = 1
        self.image1[0:10, 10] = 2
        self.image1[15, 3:8] = 1

        self.image2 = np.zeros((20, 20), dtype=np.uint8)
        self.image2[0:10, 0:12] = 3
        self.image2[15, 15] = 1
        self.image2[15, 16] = 2
        self.image2[15, 3:8] = 4

    def make_volume(self, image):
        one_slice = np.expand_dims(image, 0)
        return np.concatenate((one_slice, one_slice), axis=0)

    def test_label_cells_by_layers_smoke(self):
        def label_cell(v):
            return label_cells_cc(v, 0.5, 1, 0)

        labels = label_cells_by_layers(self.image2, label_cell, 2)
        self.assertEqual(self.image2.shape, labels.shape)

    def test_watershed_labels_70k(self):
        def label_watershed(v):
            return label_cells_watershed(v, 0.5, median_filtering=1,
                                         peak_suppress=1, smoothing=1.0,
                                         dilation=0)

        great_image = np.zeros((5, 1000, 1000), dtype=np.uint8)
        for i in range(70000):
            z = i // 20000
            rest = i % 20000
            y = rest // 450 * 2 + z * 3
            x = rest % 450 * 2 + z * 3
            great_image[z, y, x] = 130

        labels = label_cells_by_layers(great_image, label_watershed, 2)
        self.assertEqual(great_image.shape, labels.shape)
        self.assertEqual(np.uint32, labels.dtype)
        nptest.assert_equal(labels > 0, great_image > 0)
        self.assertEqual(70000 + 1, len(np.unique(labels)))

    def test_consistent_labels(self):
        vol0 = self.make_volume(self.image1)
        vol1 = self.make_volume(self.image1)
        vol2 = self.make_volume(self.image2)

        vol1_consist, vol2_consist = make_consistent_labels(vol1, vol2)
        image1_consist = vol1_consist[0]
        image2_consist = vol2_consist[0]

        self.assertEqual(1, image1_consist[5, 6])
        self.assertEqual(1, image1_consist[5, 10])
        self.assertEqual(2, vol1[0][5, 10])
        self.assertEqual(1, image1_consist[15, 5])

        self.assertEqual(1, image2_consist[5, 6])
        self.assertEqual(3, image2_consist[15, 15])
        self.assertEqual(4, image2_consist[15, 16])
        self.assertEqual(1, image2_consist[15, 5])

        # check if in place method works
        make_consistent_labels([vol0, vol1], vol2, return_copy=False)
        nptest.assert_array_equal(vol0, vol1_consist)
        nptest.assert_array_equal(vol1, vol1_consist)
        nptest.assert_array_equal(vol2, vol2_consist)

    def test_find_corresponding_labels(self):
        mapping = find_corresponding_labels(self.image1, self.image2)
        self.assertEqual(True, 0 not in mapping)
        self.assertEqual(3, mapping[1])
        self.assertEqual(3, mapping[2])
        self.assertEqual(True, 3 not in mapping)

    def test_dilation_only_2d(self):
        labels_volume = np.zeros((3, 20, 20), dtype=np.uint8)
        labels_volume[0][0:10, 0:10] = 1
        labels_volume[1][5:6, 5:6] = 1
        labels_volume[2][0:10, 0:10] = 1

        labels_dilated = dilation_only_2d(labels_volume, 1)
        expected_middle = np.zeros((20, 20), dtype=np.uint8)
        expected_middle[4:7, 4:7] = 1
        expected_middle[4, 4] = 0
        expected_middle[4, 6] = 0
        expected_middle[6, 4] = 0
        expected_middle[6, 6] = 0
        nptest.assert_array_equal(expected_middle, labels_dilated[1])

    def tearDown(self):
        pass
