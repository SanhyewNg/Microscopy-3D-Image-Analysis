import pytest
import unittest

import numpy as np
import numpy.testing as nptest

from clb.volume_slicer import VolumeSlicer


@pytest.mark.preprocessing
class TestVolumeSlicer(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

    def test_restiching(self):
        random_volume = np.random.random((20, 60, 60))
        volume_slicer1 = VolumeSlicer(4)
        volume_slicer2 = VolumeSlicer(7)
        slicer1 = volume_slicer1.divide_volume(random_volume)
        slicer2 = volume_slicer2.divide_volume(random_volume)
        self.assertNotEqual(len(slicer1), len(slicer2))

        stitch1 = volume_slicer1.stitch_volume(slicer1)
        stitch2 = volume_slicer2.stitch_volume(slicer2)
        stitch2_by1 = volume_slicer1.stitch_volume(slicer2)
        nptest.assert_array_almost_equal(random_volume, stitch1)
        nptest.assert_array_almost_equal(random_volume, stitch2)
        nptest.assert_array_almost_equal(stitch1, stitch2)
        nptest.assert_array_almost_equal(stitch2_by1, stitch2)

    def tearDown(self):
        pass
