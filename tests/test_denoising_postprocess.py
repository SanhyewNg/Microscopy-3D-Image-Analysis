import pytest
import unittest

import numpy as np

import clb.denoising.postprocess as postproc


@pytest.mark.denoising
class TestPostprocessFunctions(unittest.TestCase):
    def test_if_merge_patches_gives_right_output(self):
        patches = np.array([
            [
                [1, 2],
                [4, 5]
            ],
            [
                [2, 3],
                [5, 6]
            ],
            [
                [4, 5],
                [7, 8]
            ],
            [
                [5, 6],
                [8, 9]
            ],
        ], dtype=np.float64).reshape((4, 2, 2))

        image = postproc.merge_patches(patches, (3, 3), stride=(1, 1))
        expected_image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        np.testing.assert_equal(image, expected_image)
