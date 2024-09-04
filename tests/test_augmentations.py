import copy
import pytest
from unittest import TestCase

import numpy as np
import numpy.testing as nptest
import scipy.ndimage.interpolation as sni

from clb.dataprep.augmenter.augmentations import (Flip, Rotation, Scale, Shift)


@pytest.mark.preprocessing
class TestAugmentations(TestCase):

    def setUp(self):
        np.random.seed(10)
        self.image_random_3d = np.random.random((256, 256, 256)) * 256
        self.image_3d = np.array([[[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]],
                                  [[16, 17, 18, 19],
                                   [20, 21, 22, 23],
                                   [24, 25, 26, 27],
                                   [13, 14, 15, 16]],
                                  [[17, 18, 19, 20],
                                   [21, 22, 23, 24],
                                   [25, 26, 27, 28],
                                   [29, 30, 31, 32]],
                                  [[33, 34, 35, 36],
                                   [37, 38, 39, 40],
                                   [41, 42, 43, 44],
                                   [45, 46, 47, 48]]])

    def test_augmentation_probability_edge_case(self):

        flip_false = Flip(probability=0.0, axis=0)
        flip_true = Flip(probability=1.0, axis=0)

        flipped_image = flip_true.augment(self.image_3d)
        non_flipped_image = flip_false.augment(self.image_3d)
        nptest.assert_array_equal(flipped_image, np.flip(self.image_3d, axis=0))
        nptest.assert_array_equal(non_flipped_image, self.image_3d)
        # TODO: switch flip to mock

    def test_algorithm_called_with_correct_params(self):
        pass
        #TODO: TESTS!!
