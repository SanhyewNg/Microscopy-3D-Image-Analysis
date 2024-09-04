import pytest
import unittest

import numpy as np
import numpy.testing as nptest
import scipy.ndimage

import clb.classify.cell_extractor as cell_extractor
import clb.classify.extractors as extractors
import clb.classify.utils as utils


@pytest.mark.classification
class TestCellExtractor(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

        self.intensity = np.random.random((30, 50, 50, 3))
        self.labels = np.zeros((30, 50, 50), dtype=np.uint8)

    def test_extract_cells_crops_single_no_rescale_inside(self):
        labels = self.labels.copy()
        labels[5:8, 7:10, 7:10] = 1

        cell_crops = cell_extractor.extract_cells_crops(self.intensity, labels, crop_size=0.5)
        self.assertEqual(True, 1 in cell_crops)
        self.assertEqual(1, len(cell_crops))

        one_crop = cell_crops[1]['input']
        one_label_crop = cell_crops[1]['contour']
        self.assertEqual((1, 1, 1, 3), one_crop.shape)
        self.assertEqual((1, 1, 1), one_label_crop.shape)
        nptest.assert_array_equal(self.intensity[6:7, 8:9, 8:9], one_crop)
        nptest.assert_array_equal(1, one_label_crop[0, 0, 0])

        cell_crops = cell_extractor.extract_cells_crops(self.intensity, labels, crop_size=2.5)
        one_crop = cell_crops[1]['input']
        one_label_crop = cell_crops[1]['contour']
        self.assertEqual((5, 5, 5, 3), one_crop.shape)
        self.assertEqual((5, 5, 5), one_label_crop.shape)
        nptest.assert_array_equal(self.intensity[4:9, 6:11, 6:11], one_crop)
        nptest.assert_array_equal(1, one_label_crop[2, 2, 2])
        nptest.assert_array_equal(3 * 3 * 3, one_label_crop.sum())

    def test_extract_cells_crops_single_rescale_inside(self):
        labels = self.labels.copy()
        labels[5:8, 7:10, 7:10] = 10

        # just one pixel but coming from multiple area (imagery is zoommed in)
        cell_crops = cell_extractor.extract_cells_crops(self.intensity, labels, crop_size=0.5,
                                                        voxel_size=(0.3, 0.3, 0.3))
        self.assertEqual(True, 10 in cell_crops)
        self.assertEqual(1, len(cell_crops))

        one_crop = cell_crops[10]['input']
        one_label_crop = cell_crops[10]['contour']
        self.assertEqual((1, 1, 1, 3), one_crop.shape)
        self.assertEqual((1, 1, 1), one_label_crop.shape)
        # this is actually an aggregation of 5x5x5 neighbour pixels
        self.assertNotEqual(self.intensity[6, 8, 8, 0], one_crop[0, 0, 0, 0])

        # we still want it to have that label
        nptest.assert_array_equal(10, one_label_crop[0, 0, 0])
        self.assertEqual(1, len(np.unique(one_label_crop)))

        # now for a bigger and non uniform case
        cell_crops = cell_extractor.extract_cells_crops(self.intensity, labels, crop_size=3,
                                                        voxel_size=(0.2, 0.4, 0.4))

        self.assertEqual(True, 10 in cell_crops)
        self.assertEqual(1, len(cell_crops))

        one_crop = cell_crops[10]['input']
        one_label_crop = cell_crops[10]['contour']
        self.assertEqual((6, 6, 6, 3), one_crop.shape)
        self.assertEqual((6, 6, 6), one_label_crop.shape)

        # we still want it to have that label
        nptest.assert_array_equal(10, one_label_crop[3, 3, 3])
        unique_labels = list(np.unique(one_label_crop))
        self.assertEqual([0, 10], unique_labels)

    def test_extract_cells_crops_single_no_rescale_border(self):
        # pixel center is (1,49,6) it is 3x3x3 block
        labels = self.labels.copy()
        labels[0:3, 47:51, 5:8] = 10

        # 5x5 pixels
        cell_crops = cell_extractor.extract_cells_crops(self.intensity, labels, crop_size=2.5)
        self.assertEqual(True, 10 in cell_crops)
        self.assertEqual(1, len(cell_crops))

        one_crop = cell_crops[10]['input']
        one_label_crop = cell_crops[10]['contour']
        self.assertEqual((5, 5, 5, 3), one_crop.shape)
        self.assertEqual((5, 5, 5), one_label_crop.shape)
        nptest.assert_array_equal(self.intensity[0:4, 46:51, 4:9], one_crop[1:5, 0:4, 0:5])

        # we still want it to have that label
        nptest.assert_array_equal(10, one_label_crop[2, 2, 2])
        unique_labels = list(np.unique(one_label_crop))
        self.assertEqual([0, 10], unique_labels)

    def test_extract_cells_crops_two_close_rescale_zoom_out(self):
        # two cells 3x6x6 touching on y axis
        labels = self.labels.copy()
        labels[13:16, 11:17, 11:17] = 10
        labels[13:16, 17:23, 11:17] = 20

        # x and y should get 2x smaller, 10x10x10 cube
        cell_crops = cell_extractor.extract_cells_crops(self.intensity, labels,
                                                        crop_size=5, voxel_size=(0.5, 0.25, 0.25))
        self.assertEqual(True, 10 in cell_crops)
        self.assertEqual(True, 20 in cell_crops)
        self.assertEqual(2, len(cell_crops))
        one_crop = cell_crops[10]['input']
        one_label_crop = cell_crops[10]['contour']
        two_crop = cell_crops[20]['input']
        two_label_crop = cell_crops[20]['contour']
        self.assertEqual((10, 10, 10, 3), one_crop.shape)
        self.assertEqual((10, 10, 10, 3), two_crop.shape)
        self.assertEqual((10, 10, 10), one_label_crop.shape)
        self.assertEqual((10, 10, 10), two_label_crop.shape)

        # same labels on each 3 slices
        for z in range(4, 7):
            nptest.assert_array_equal(one_label_crop[4], one_label_crop[z])
            nptest.assert_array_equal(two_label_crop[4], two_label_crop[z])

        # each crop should contain both cells
        self.assertEqual([0, 10, 20], list(np.unique(one_label_crop)))
        self.assertEqual([0, 10, 20], list(np.unique(two_label_crop)))
        self.assertEqual(3 * (3 * 3 * 10 + 3 * 3 * 20), one_label_crop.sum())
        self.assertEqual(3 * (3 * 3 * 10 + 3 * 3 * 20), two_label_crop.sum())

    def test_extract_cells_crops_two_close_rescale_zoom_in(self):
        # two cells 2x4x4 touching on y axis
        labels = self.labels.copy()
        labels[14:16, 10:14, 20:24] = 10
        labels[14:16, 14:18, 20:24] = 20

        # x and y should get 2x bigger, 10x10x10 cube
        cell_crops = cell_extractor.extract_cells_crops(self.intensity, labels,
                                                        crop_size=5, voxel_size=(0.5, 1, 1))
        self.assertEqual(True, 10 in cell_crops)
        self.assertEqual(True, 20 in cell_crops)
        self.assertEqual(2, len(cell_crops))
        one_crop = cell_crops[10]['input']
        one_label_crop = cell_crops[10]['contour']
        two_crop = cell_crops[20]['input']
        two_label_crop = cell_crops[20]['contour']
        self.assertEqual((10, 10, 10, 3), one_crop.shape)
        self.assertEqual((10, 10, 10, 3), two_crop.shape)
        self.assertEqual((10, 10, 10), one_label_crop.shape)
        self.assertEqual((10, 10, 10), two_label_crop.shape)

        # each crop should contain both cells
        self.assertEqual([0, 10], list(np.unique(one_label_crop)))
        self.assertEqual([0, 10, 20], list(np.unique(two_label_crop)))  # contain also part of 10
        self.assertEqual(2 * (8 * 8 * 10), one_label_crop.sum())
        self.assertEqual(2 * (2 * 8 * 10 + 8 * 8 * 20), two_label_crop.sum())

    def test_extract_cells_crops_object_in_center_rescale_zoom_in(self):
        labels = np.zeros((3, 100, 100), dtype=np.uint8)
        labels[0:2, 10:30, 20:50] = 10

        # x and y should get 2x bigger, 70x70x70 cube
        # cell should be 2 x 40 x 60 centered
        cell_crops = cell_extractor.extract_cells_crops(labels, labels,
                                                        crop_size=35, voxel_size=(0.5, 1, 1))
        one_label_crop = cell_crops[10]['contour']
        self.assertEqual([0, 10], list(np.unique(one_label_crop)))
        self.assertEqual(2 * (40 * 60 * 10), one_label_crop.sum())

        # test if it is in the center (approx.)
        center = scipy.ndimage.center_of_mass(one_label_crop)
        self.assertAlmostEqual(center[0], one_label_crop.shape[0] / 2, delta=0.5)
        self.assertAlmostEqual(center[1], one_label_crop.shape[1] / 2, delta=0.5)

    def test_extract_cells_crops_object_in_center_rescale_zoom_out(self):
        labels = np.zeros((3, 100, 100), dtype=np.uint8)
        labels[0:2, 10:30, 20:50] = 10

        # x and y should get 2x smaller, 60x60x60 cube
        # cell should be 2 x 10 x 15 centered
        cell_crops = cell_extractor.extract_cells_crops(labels, labels,
                                                        crop_size=30, voxel_size=(0.5, 0.25, 0.25))
        one_label_crop = cell_crops[10]['contour']
        self.assertEqual([0, 10], list(np.unique(one_label_crop)))

        # should have 150 pixels
        self.assertAlmostEqual(2 * (10 * 15 * 10), one_label_crop.sum(), delta=10 * 30)  # 20%

        # test if it is in the center (approx.)
        center = scipy.ndimage.center_of_mass(one_label_crop)
        self.assertAlmostEqual(center[0], one_label_crop.shape[0] / 2, delta=0.5)
        self.assertAlmostEqual(center[1], one_label_crop.shape[1] / 2, delta=0.5)

    def tearDown(self):
        pass
