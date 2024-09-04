import pytest
import itertools as it
import unittest
import unittest.mock as mock

import numpy as np
import tensorflow as tf

import clb.denoising.preprocess as preproc
import tests.denoising_test_utils as tutils


@pytest.mark.denoising
class TestAugment(unittest.TestCase):
    @staticmethod
    def to_tuple(array):
        return tuple(array.flat)

    def test_if_gives_right_output(self):
        image = np.array([
            [1, 2],
            [3, 4]
        ]).reshape(2, 2, 1)

        # I wanted to check if sets are equal, but since arrays aren't hashable
        # I'm changing them to 4-tuples.
        augmented = {
            self.to_tuple(img)
            for img in preproc.augment(image)
        }
        expected = {
            self.to_tuple(np.array([
                [1, 2],
                [3, 4]
            ])),
            self.to_tuple(np.array([
                [2, 4],
                [1, 3]
            ])),
            self.to_tuple(np.array([
                [4, 3],
                [2, 1]
            ])),
            self.to_tuple(np.array([
                [3, 1],
                [4, 2]
            ])),
            self.to_tuple(np.array([
                [3, 4],
                [1, 2]
            ])),
            self.to_tuple(np.array([
                [4, 2],
                [3, 1]
            ])),
            self.to_tuple(np.array([
                [2, 1],
                [4, 3]
            ])),
            self.to_tuple(np.array([
                [1, 3],
                [2, 4]
            ])),
        }

        self.assertEqual(augmented, expected)


class TestAddNoise(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.addCleanup(self.sess.close)

    def test_if_noise_is_added(self):
        image = tf.constant([
            [0.1, 0.2],
            [0.3, 0.4]
        ])
        noise = tf.constant([
            [0.4, 0.3],
            [0.2, 0.1]
        ])
        noiser = mock.Mock(return_value=noise)

        noised = preproc.add_noise(image, noiser=noiser)
        noised = self.sess.run(noised)
        expected = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])

        np.testing.assert_equal(noised, expected)

    def test_if_noiser_is_called_with_right_arguments(self):
        image = tf.constant([
            [0.1, 0.2],
            [0.3, 0.4]
        ])
        noise = tf.constant([
            [0.4, 0.3],
            [0.2, 0.1]
        ])
        noiser = mock.Mock(return_value=noise)

        preproc.add_noise(image, noiser=noiser)

        noiser.assert_called_once_with(
            dtype=image.dtype,
            shape=tutils.Tensor(self.sess, image.shape)
        )

    def test_if_add_noise_clips_output(self):
        image = tf.constant([
            [0.6, 0.7],
            [0.8, 0.1]
        ])
        noise = tf.constant([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        noiser = mock.Mock(return_value=noise)

        noised = preproc.add_noise(image, noiser=noiser)
        noised = self.sess.run(noised)
        expected = np.array([
            [1., 1.],
            [1., 0.6]
        ], dtype=np.float32)

        np.testing.assert_equal(noised, expected)


class TestExtractPatches(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.addCleanup(self.sess.close)

    def test_if_extract_patches_gives_right_output_without_padding(self):
        image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        patches = preproc.extract_patches(image,
                                          shape=(2, 2),
                                          stride=(1, 1))

        expected_patches = np.array([
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
            ]
        ]).reshape((4, 2, 2))

        np.testing.assert_equal(patches, expected_patches)

    def test_if_extract_patches_gives_right_output_with_padding(self):
        image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        patches = preproc.extract_patches(image,
                                          shape=(2, 2),
                                          stride=(1, 1))

        expected_patches = np.array([
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
        ]).reshape((4, 2, 2))

        np.testing.assert_equal(patches, expected_patches)
