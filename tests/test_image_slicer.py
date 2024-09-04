import pytest
import unittest

import numpy as np
import numpy.testing as nptest

from clb.image_slicer import ImageSlicer


@pytest.mark.io
class TestImageSlicer(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

    @pytest.mark.skip("To fix later")
    def test_restiching(self):
        random_image = np.random.random((112, 112))
        slicer_small = ImageSlicer(40, 40, 10)
        slicer_big = ImageSlicer(140, 140, 30)
        slices_small = slicer_small.divide_image(random_image)
        slices_big = slicer_big.divide_image(random_image)
        self.assertEqual(9, len(slices_small))
        self.assertEqual(1, len(slices_big))
        self.assertEqual(random_image.shape, slices_big[0])

        stitched_small = slicer_small.stitch_images(slices_small)
        stitched_big = slicer_big.stitch_images(slices_big)
        nptest.assert_array_almost_equal(random_image, stitched_small)
        nptest.assert_array_almost_equal(random_image, stitched_big)
        nptest.assert_array_almost_equal(stitched_small, stitched_big)

    def tearDown(self):
        pass
