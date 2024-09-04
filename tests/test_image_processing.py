import pytest
import unittest

import numpy as np
import numpy.testing as nptest
from skimage import measure

from clb.image_processing import (remove_annotated_blobs,
                                   remove_annotated_blobs_when_overlap,
                                   find_corresponding_labels, clahe, extend_membrane, estimate_membrane_from_nucleus,
                                   resample)
from tests.utils import get_random_labels


@pytest.mark.io
class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

    def test_remove_annotated_blobs(self):
        image = np.ones((20, 20))
        image[5:15, 5:8] = 10
        image[2, 13] = 20
        label = np.zeros((20, 20))
        label[5:10, 5:8] = 1
        label[2, 13] = 2
        label[10:13, 5:8] = 3

        image_with_removed_blobs = remove_annotated_blobs(image, label)
        label_with_removed_blobs = remove_annotated_blobs(label, label)
        self.assertEqual(0, image_with_removed_blobs[5, 5])
        self.assertEqual(0, image_with_removed_blobs[9, 7])
        self.assertEqual(0, image_with_removed_blobs[10, 7])
        self.assertEqual(10, image_with_removed_blobs[11, 7])
        self.assertEqual(20, image_with_removed_blobs[2, 13])

        self.assertEqual(0, label_with_removed_blobs[5, 5])
        self.assertEqual(0, label_with_removed_blobs[9, 7])
        self.assertEqual(0, label_with_removed_blobs[10, 7])
        self.assertEqual(2, label_with_removed_blobs[2, 13])
        self.assertEqual(3, label_with_removed_blobs[12, 7])

    def test_remove_annotated_blobs_with_partial(self):
        image = np.zeros((20, 20))
        image[5:10, 5:10] = 5  # entire in blob
        image[2:3, 3:5] = 6  # outside of blob
        image[5:7, 3:7] = 7  # half in the blob
        label = np.zeros((20, 20))
        label[5:15, 5:15] = 1

        image_with_removed_blobs = remove_annotated_blobs(image, label)
        self.assertEqual(False, (image_with_removed_blobs == 5).any())
        self.assertEqual(True, (image_with_removed_blobs == 6).any())
        self.assertEqual(True, (image_with_removed_blobs == 7).any())

        image_with_removed_blobs_when_overlap = remove_annotated_blobs_when_overlap(image, label)
        self.assertEqual(False, (image_with_removed_blobs_when_overlap == 5).any())
        self.assertEqual(True, (image_with_removed_blobs_when_overlap == 6).any())
        self.assertEqual(False, (image_with_removed_blobs_when_overlap == 7).any())

    def test_remove_annotated_blobs_massive_undersegmentation(self):
        image = np.zeros((20, 20))
        image[5:15, 5:15] = 5  # covers multiple cells but also blob
        image[17:20, 6:13] = 6  # covers multiple cells but also blob
        label = np.zeros((20, 20))
        label[5:10, 5:10] = 1
        label[10:13, 5:10] = 2
        label[5:10, 10:13] = 3

        image_with_removed_blobs_when_overlap = remove_annotated_blobs_when_overlap(image, label,
                                                                                    minimum_blob_overlap=0)
        self.assertEqual(False, (image_with_removed_blobs_when_overlap == 5).any())
        self.assertEqual(True, (image_with_removed_blobs_when_overlap == 6).any())

        image_with_removed_blobs_when_overlap = remove_annotated_blobs_when_overlap(image, label,
                                                                                    minimum_blob_overlap=0.4)
        self.assertEqual(True, (image_with_removed_blobs_when_overlap == 5).any())
        self.assertEqual(True, (image_with_removed_blobs_when_overlap == 6).any())

    def test_remove_annotated_blobs_giant_blob(self):
        image = np.zeros((20, 20))
        image[5:10, 5:10] = 5  # small object inside a blob
        label = np.zeros((20, 20))
        label[1:17, 3:16] = 1

        image_with_removed_blobs_when_overlap = remove_annotated_blobs_when_overlap(image, label)
        self.assertEqual(False, (image_with_removed_blobs_when_overlap == 5).any())

    def test_find_correspondence_2d(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[2:5, 3:5] = 1
        image[3, 4:6] = 2
        classes = np.zeros((10, 10), dtype=np.uint8)
        classes[:] = 3

        mapping = find_corresponding_labels(image, classes)
        self.assertTrue(0 not in mapping)
        self.assertEqual(3, mapping[1])
        self.assertEqual(3, mapping[2])
        self.assertTrue(3 not in mapping)

        mapping = find_corresponding_labels(image, classes, True, True)
        self.assertEqual((3, 1, 5), mapping[1])
        self.assertEqual((3, 1, 2), mapping[2])

        classes[:] = 0
        classes[2:4, 3:5] = 1
        classes[4, 4] = 2
        mapping = find_corresponding_labels(image, classes, True, True)
        self.assertTrue(0 not in mapping)
        self.assertEqual((1, 3 / 5, 3), mapping[1])
        self.assertEqual((1, 1 / 2, 1), mapping[2])
        self.assertTrue(3 not in mapping)

        classes[2:4, 3:5] = 0
        mapping = find_corresponding_labels(image, classes, True, True)
        self.assertEqual((2, 1 / 5, 1), mapping[1])
        self.assertTrue(2 not in mapping)

    def test_find_correspondence_3d(self):
        image = np.zeros((10, 10, 10), dtype=np.uint8)
        image[3:6, 2:5, 3:5] = 1

        classes = np.zeros((10, 10, 10), dtype=np.uint8)
        classes[4, 2:5, 3:4] = 2
        classes[5, ...] = 3

        mapping = find_corresponding_labels(image, classes, True, True)
        self.assertTrue(0 not in mapping)
        self.assertEqual((3, 1 / 3, 3 * 2), mapping[1])
        self.assertTrue(2 not in mapping)

    def test_clahe(self):
        image = np.random.rand(100, 100) / 5
        self.assertEqual(np.float64, image.dtype)
        self.assertAlmostEqual(0.2, image.max(), 3)
        self.assertAlmostEqual(0.0, image.min(), 3)

        clahe_image = clahe(image, 20, 3)
        self.assertEqual(np.float32, clahe_image.dtype)
        self.assertAlmostEqual(1.0, clahe_image.max(), 3)
        self.assertAlmostEqual(0.0, clahe_image.min(), 3)

    def test_extend_membrane(self):
        image = np.random.rand(100, 100) / 5
        self.assertEqual(np.float64, image.dtype)
        self.assertAlmostEqual(0.2, image.max(), 3)
        self.assertAlmostEqual(0.0, image.min(), 3)

        membrane_image = extend_membrane(image)
        self.assertEqual(np.float32, membrane_image.dtype)
        self.assertGreater(image.max(), membrane_image.max())
        self.assertLess(image.min(), membrane_image.min())

    def test_estimate_membrane_from_nucleus(self):
        labels = get_random_labels((100, 100), 10, 5)
        self.assertEqual(np.uint8, labels.dtype)

        membrane_labels = estimate_membrane_from_nucleus(labels)
        self.assertEqual(np.uint8, membrane_labels.dtype)

    def test_resample(self):
        def bbox_size(mask):
            cell_prop = measure.regionprops(mask, mask > 0)[0]
            sz, sy, sx, ez, ey, ex = cell_prop.bbox
            return ez - sz, ey - sy, ex - sx

        flat_disc = np.zeros((30, 30, 30), dtype=np.uint8)
        flat_disc[1:3, 2:12, 0:30] = 1

        size_org = bbox_size(flat_disc)
        self.assertEqual((2, 10, 30), size_org)

        _, r = resample(flat_disc, pixel_size=(1, 1, 1), new_pixel_size=(1, 1, 1))
        self.assertEqual(None, r)

        sample_real_case, r = resample(flat_disc, pixel_size=(1, 0.57, 0.57), new_pixel_size=(0.5, 0.5, 0.5))
        nptest.assert_almost_equal((0.5, 0.5, 0.5), r, 2)
        size_real = bbox_size(sample_real_case)
        self.assertEqual((4, 12, 34), size_real)

        upscaled, r = resample(flat_disc, pixel_size=(2, 2, 2), new_pixel_size=(1, 1, 1))
        nptest.assert_almost_equal((1, 1, 1), r, 2)
        size_upscaled = bbox_size(upscaled)
        self.assertEqual((4, 20, 60), size_upscaled)

        downscaled, r = resample(flat_disc, pixel_size=(0.5, 0.5, 0.5), new_pixel_size=(1, 1, 1))
        nptest.assert_almost_equal((1, 1, 1), r, 2)
        size_downscaled = bbox_size(downscaled)
        self.assertEqual((1, 5, 15), size_downscaled)

        only_disc = flat_disc[1:3,2:12,0:30]
        too_small, r = resample(only_disc, pixel_size=(0.1, 0.5, 0.5), new_pixel_size=(1, 1, 1))
        size_small = bbox_size(too_small)
        self.assertEqual((2, 5, 15), size_small)
        nptest.assert_almost_equal((0.1, 1, 1), r, 2)

    def tearDown(self):
        pass
