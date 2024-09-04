import pytest
import unittest

import numpy as np
import numpy.testing as nptest

from clb.predict.multiple import MultiTransformer


@pytest.mark.preprocessing
class TestPredictMultiple(unittest.TestCase):
    def test_multi_transformer(self):
        rgb_images = (np.random.random((3, 20, 20, 3)) * 250)
        gray_images = (np.random.random((3, 20, 20)) * 250)

        transformer = MultiTransformer(use_flips=True, use_rotation=True)

        rgb_trans = list(transformer.generate_transformations(rgb_images))
        self.assertEqual(len(rgb_trans), 8 * 3)

        gray_trans = list(transformer.generate_transformations(gray_images))
        self.assertEqual(len(gray_trans), 8 * 3)

        redone_rgb = list(transformer.merge_transformations(rgb_trans))
        nptest.assert_almost_equal(redone_rgb, rgb_images)

        redone_gray = list(transformer.merge_transformations(gray_trans))
        nptest.assert_almost_equal(redone_gray, gray_images)
