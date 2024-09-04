import pytest
import unittest
from collections import Counter

import numpy as np
import numpy.testing as nptest

from tests.utils import *
from clb.classify.instance_matching import cell_level_from_contours, cell_level_classes, remove_gaps_in_slices


@pytest.mark.postprocessing
class TestInstanceMatching(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

    def verify_match(self, expected, calculated):
        self.assertEqual(expected, {k: calculated[k] for k in ("id", "class")})

    def test_instance_matching_regression(self):
        np.random.seed(10)
        random.seed(10)
        labels = get_random_labels((6, 200, 200), 200)
        classes = get_random_classes((6, 200, 200))

        label_classes = cell_level_from_contours(labels, classes, 0.45)

        counter = Counter([v['class'] for k, v in label_classes.items()])
        self.assertEqual(161, counter[0])
        self.assertEqual(23, counter[1])
        self.assertEqual(16, counter[2])

    def test_cell_level_from_contours(self):
        labels = np.zeros((3, 50, 50), dtype=np.uint8)
        labels[0, 3:10, 3:10] = 1
        labels[0, 12:16, 12:16] = 2

        # cell in two layers but mostly on second
        labels[1, 5:10, 2:9] = 3
        labels[2, 5:8, 2:5] = 3

        labels[2, 1:6, 10:15] = 4

        # not marked cell
        labels[1, 1:3, 1:3] = 7

        classes = np.zeros((3, 50, 50), dtype=np.uint8)
        classes[0, 5:20, 5:15] = 1
        classes[1, 4:12, 4:8] = 2
        classes[2] = 3

        cell_classes = cell_level_from_contours(labels, classes, 0.45)

        self.assertEqual(False, 0 in cell_classes)
        self.verify_match({"id": 1, "class": 1}, cell_classes[1])
        self.verify_match({"id": 2, "class": 1}, cell_classes[2])
        self.verify_match({"id": 3, "class": 2}, cell_classes[3])
        self.verify_match({"id": 4, "class": 3}, cell_classes[4])
        self.assertEqual(False, 5 in cell_classes)
        self.assertEqual(False, 6 in cell_classes)
        self.verify_match({"id": 7, "class": 0}, cell_classes[7])

    def test_cell_level_with_partial_classes(self):
        labels = np.zeros((3, 30, 30), dtype=np.uint8)

        labels[0, 3:10, 3:10] = 1
        labels[1, 5:7, 5:7] = 1
        labels[2, 3:10, 3:10] = 1

        labels[0, 5:7, 5:7] = 2
        labels[1, 10:20, 10:20] = 2
        labels[2, 5:7, 5:7] = 2

        labels[2, 20:23, 20:23] = 3
        labels[1, 0:2, 0:2] = 4

        classes = np.zeros((3, 30, 30), dtype=np.uint8)
        classes[0, 1:2, 1:2] = 1  # just to make it not empty
        classes[1, 4:32, 4:32] = 2
        classes[2, 1:2, 1:2] = 1

        cell_classes = cell_level_classes(labels, classes, "tissue", 0.45)
        self.assertEqual(False, 0 in cell_classes)
        # 1 is mostly zeroes
        self.verify_match({"id": 1, "class": 0}, cell_classes[1])
        self.verify_match({"id": 2, "class": 2}, cell_classes[2])
        self.verify_match({"id": 3, "class": 0}, cell_classes[3])

        # make it partial so only middle slice counts
        classes[0] = 0
        classes[2] = 0

        cell_classes = cell_level_classes(labels, classes, "tissue", 0.45)
        self.assertEqual(False, 0 in cell_classes)
        self.verify_match({"id": 1, "class": 2}, cell_classes[1])
        self.verify_match({"id": 2, "class": 2}, cell_classes[2])
        self.assertEqual(False, 3 in cell_classes)  # cell is outside of class annotation
        self.verify_match({"id": 4, "class": 0}, cell_classes[4])

    def test_cell_level_with_empty(self):
        labels = np.zeros((4, 30, 30), dtype=np.uint8)

        labels[0, 3:10, 3:10] = 1
        labels[1, 5:7, 5:7] = 1
        labels[2, 5:7, 5:7] = 1
        labels[3, 3:10, 3:10] = 1

        labels[0, 15:7, 15:7] = 5
        labels[1, 5:7, 5:7] = 3
        labels[2, 10:20, 10:20] = 2
        labels[3, 5:7, 5:7] = 3
        labels[3, 15:7, 15:7] = 4

        classes = np.zeros((4, 30, 30), dtype=np.uint8)
        cell_classes = cell_level_classes(labels, classes, "tissue", 0.45)

        # only middle slice are annotated
        self.assertEqual(False, 0 in cell_classes)
        self.verify_match({"id": 1, "class": 0}, cell_classes[1])
        self.verify_match({"id": 2, "class": 0}, cell_classes[2])
        self.assertEqual(False, 3 in cell_classes)
        self.assertEqual(False, 4 in cell_classes)

    def test_remove_gaps(self):
        labels = np.zeros((5, 30, 30), dtype=np.uint8)
        labels[0, 3:10, 3:10] = 1
        labels[1, 4:7, 5:7] = 2
        labels[2, 5:7, 4:7] = 3
        labels[3, 5:7, 4:7] = 4
        labels[4] = 0

        removed = remove_gaps_in_slices(labels.copy())
        # nothing changes because there are no
        nptest.assert_array_equal(labels, removed)

        labels[1] = 0
        removed = remove_gaps_in_slices(labels.copy())
        # the existing slices are the same
        nptest.assert_array_equal(labels[0], removed[0])
        nptest.assert_array_equal(labels[2], removed[2])
        nptest.assert_array_equal(labels[3], removed[3])
        # the trailing empty slice is changed
        self.assertEqual(True, np.any(removed[4]))

        # in filled slice both values are present
        self.assertEqual([0, 1, 3], list(np.unique(removed[1])))

    def test_cell_level_with_label_gaps(self):
        labels = np.zeros((4, 30, 30), dtype=np.uint8)

        labels[1, 5:7, 5:7] = 1
        labels[3, 3:10, 3:10] = 1

        classes = np.zeros((4, 30, 30), dtype=np.uint8)
        classes[2, 3:9, 3:9] = 1

        cell_classes = cell_level_classes(labels, classes, "tissue", fill_gaps=False, overlap_threshold=0.45)
        # nothing is matched as slices do not ovelap
        self.assertEqual(False, 0 in cell_classes)
        self.assertEqual(False, 1 in cell_classes)

        cell_classes = cell_level_classes(labels, classes, "tissue", fill_gaps=True, overlap_threshold=0.45)
        # annotation for unannotated slice are taken from neighbouring slices
        self.verify_match({"id": 1, "class": 1}, cell_classes[1])

    def tearDown(self):
        pass
