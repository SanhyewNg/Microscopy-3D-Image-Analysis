import pytest
import unittest

from unittest.mock import patch
import numpy as np
import numpy.testing as nptest

from clb.classify.visualisation import ClassificationVolume


@pytest.mark.io
class TestClassificationVolume(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.labels_volume = np.zeros((5, 60, 60))
        self.labels_volume[1:3, 5:8, 3:6] = 1
        self.labels_volume[4, 5:8, 3:6] = 2
        self.labels_volume[2:5, 10:15, 20:30] = 3
        self.labels_volume[0, 3, 4] = 4

    def test_none(self):
        overlay = ClassificationVolume.create(self.labels_volume, {})
        self.assertEqual([0], np.unique(overlay))

    def test_all_negative(self):
        overlay = ClassificationVolume.create(self.labels_volume, {i: 0.0 for i in range(1, 4)})
        nptest.assert_array_equal([0, 2], np.unique(overlay))

    def test_all_positive(self):
        overlay = ClassificationVolume.create(self.labels_volume, {i: 1.0 for i in range(1, 4)})
        nptest.assert_array_equal([0, 255], np.unique(overlay))

    def test_binary(self):
        binary_class_mapping = {1: 0, 2: 1, 3: 0}
        overlay = ClassificationVolume.create(self.labels_volume, binary_class_mapping)
        nptest.assert_array_equal([0, 2, 255], np.unique(overlay))

        nptest.assert_equal(overlay[1:3, 5:8, 3:6], 2)
        nptest.assert_equal(overlay[4, 5:8, 3:6], 255)
        nptest.assert_equal(overlay[2:5, 10:15, 20:30], 2)
        nptest.assert_equal(overlay[0, 3, 4], 0)

        overlay_no_rescale = ClassificationVolume.create(self.labels_volume, binary_class_mapping, rescale=False)
        nptest.assert_array_equal(overlay, overlay_no_rescale)

    def test_probs(self):
        probs = {1: 0.2, 2: 0.7, 3: 0.5}
        overlay = ClassificationVolume.create(self.labels_volume, probs, rescale=True)
        nptest.assert_array_equal([0, 2, 153, 255], np.unique(overlay))

        nptest.assert_equal(overlay[1:3, 5:8, 3:6], 2)
        nptest.assert_equal(overlay[4, 5:8, 3:6], 255)
        nptest.assert_equal(overlay[2:5, 10:15, 20:30], 153)
        nptest.assert_equal(overlay[0, 3, 4], 0)

        overlay_no_rescale = ClassificationVolume.create(self.labels_volume, probs, rescale=False)
        nptest.assert_array_equal([0, 52, 128, 179], np.unique(overlay_no_rescale))
   
    def test_validate_raises_if_ndim_other_than_2_and_3(self):
        ndim1 = np.zeros((2), dtype=np.uint8)
        ndim4 = np.zeros((2,3,4,5), dtype=np.uint8)
        with self.assertRaisesRegex(AssertionError, "Dimension of classify volume should.*"):
            ClassificationVolume._validate(ndim1)
        with self.assertRaisesRegex(AssertionError, "Dimension of classify volume should.*"):
            ClassificationVolume._validate(ndim4)

    def test_validate_min_max(self):
        over = np.full((1,2,3), 256)
        under = np.full((1,2,3), -1)
        with self.assertRaisesRegex(AssertionError, "Values of ClassificationVolume should fit.*"):
            ClassificationVolume._validate(over)
        with self.assertRaisesRegex(AssertionError, "Values of ClassificationVolume should fit.*"):
            ClassificationVolume._validate(under)

    @patch("imageio.volread")
    def test_from_file(self, volread):
        volread.return_value = self.labels_volume.astype(np.uint8)
        nptest.assert_equal(ClassificationVolume.from_file("dummy_file"), self.labels_volume)

    def test_from_array(self):
        volume = self.labels_volume.astype(np.uint8)
        nptest.assert_equal(ClassificationVolume.from_array(volume), self.labels_volume)

    def test_binary_create(self): 
        binary_class_mapping = {1: 0, 2: 1, 3: 0}
        binary_overlay = ClassificationVolume.create(self.labels_volume, binary_class_mapping, binary=True)
        
        nptest.assert_array_equal([0, 1], np.unique(binary_overlay))
        nptest.assert_equal(binary_overlay[1:3, 5:8, 3:6], 0)
        nptest.assert_equal(binary_overlay[4, 5:8, 3:6], 1)
        nptest.assert_equal(binary_overlay[2:5, 10:15, 20:30], 0)
        nptest.assert_equal(binary_overlay[0, 3, 4], 0)

    @patch("imageio.volread")
    def test_binary_from_file(self, volread): 
        probs = {1: 0.2, 2: 0.7, 3: 0.5}
        volread.return_value = ClassificationVolume.create(self.labels_volume, probs)
        overlay = ClassificationVolume.from_file("dummy_file", binary=True)
    
        nptest.assert_array_equal([0, 1], np.unique(overlay))
        nptest.assert_equal(overlay[1:3, 5:8, 3:6], 0)
        nptest.assert_equal(overlay[4, 5:8, 3:6], 1)
        nptest.assert_equal(overlay[2:5, 10:15, 20:30], 1)
        nptest.assert_equal(overlay[0, 3, 4], 0)

    def test_binary_from_array(self): 
        probs = {1: 0.2, 2: 0.7, 3: 0.5}
        overlay = ClassificationVolume.from_array(ClassificationVolume.create(self.labels_volume, probs), binary=True)

        nptest.assert_array_equal([0, 1], np.unique(overlay))
        nptest.assert_equal(overlay[1:3, 5:8, 3:6], 0)
        nptest.assert_equal(overlay[4, 5:8, 3:6], 1)
        nptest.assert_equal(overlay[2:5, 10:15, 20:30], 1)
        nptest.assert_equal(overlay[0, 3, 4], 0)

    def tearDown(self):
        pass
