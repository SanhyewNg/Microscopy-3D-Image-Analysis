import os
import pytest
import shutil
import tempfile
import unittest
import unittest.mock as mock

import imageio
import numpy as np
import numpy.testing as nptest

import clb.dataprep.readers as readers
from clb.dataprep.utils import read_volume


@pytest.mark.io
class TestReadingWriting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp()
        ones = np.ones((5, 10), dtype=np.uint8)
        twos = 2 * ones
        threes = 3 * ones
        cls.slice_path = os.path.join(cls.path, 'slice.tif')
        cls.slice = ones
        imageio.mimwrite(cls.slice_path, [cls.slice])

        cls.multich_slice_path = os.path.join(cls.path, 'multi_slice.tif')
        cls.multich_slice = np.stack((ones, twos, threes), axis=-1)
        imageio.mimwrite(cls.multich_slice_path, [cls.multich_slice])

        cls.multi_path = os.path.join(cls.path, 'multi.tif')
        cls.multi = np.array([cls.multich_slice, cls.multich_slice])
        imageio.mimwrite(cls.multi_path, [cls.multich_slice, cls.multich_slice])

        cls.first_channel = np.zeros((4, 4, 4), dtype=np.uint16)
        cls.second_channel = np.zeros((4, 4, 4), dtype=np.uint16)
        cls.third_channel = np.array([
            [[512, 2048, 1792, 1536],
             [1280, 1280, 768, 1024],
             [768, 1280, 1280, 768],
             [256, 1024, 1792, 1024]],

            [[2560, 2048, 1280, 1024],
             [1280, 1536, 256, 1280],
             [2304, 3328, 768, 1792],
             [2048, 1792, 768, 1280]],

            [[1024, 1024, 1536, 3072],
             [768, 1280, 3328, 2304],
             [1536, 1024, 2304, 2048],
             [768, 768, 768, 768]],

            [[768, 1792, 2048, 2560],
             [512, 1280, 2816, 3328],
             [768, 1024, 3328, 2048],
             [1280, 1792, 3840, 3072]]
        ], dtype=np.uint16)
        all_channels = np.stack([cls.first_channel, cls.second_channel,
                                 cls.third_channel], axis=-1)
        slices = list(all_channels)
        cls.image_path = os.path.join(cls.path, 'image.tif')
        imageio.mimwrite(cls.image_path, slices)

    def test_volread_one_slice(self):
        read_img = imageio.volread(self.slice_path)

        nptest.assert_equal(read_img, self.slice)

    def test_volread_multichannel_one_slice(self):
        read_img = imageio.volread(self.multich_slice_path)

        nptest.assert_equal(read_img, self.multich_slice)

    def test_volread_multichannel(self):
        read_img = imageio.volread(self.multi_path)

        nptest.assert_equal(read_img, self.multi)

    def test_volread_multichannel_one_slice_reading_two_channels(self):
        read_img = imageio.volread(self.multich_slice_path, channels=[0, 2])

        nptest.assert_equal(read_img, self.multich_slice[..., [0, 2]])

    def test_volumeiter_one_slice(self):
        with readers.get_volume_reader(self.slice_path) as reader:
            read_img = list(reader)

        nptest.assert_equal(np.squeeze(read_img[0]), self.slice)

    def test_volumeiter_multichannel_one_slice(self):
        with readers.get_volume_reader(self.multich_slice_path) as reader:
            read_img = list(reader)

        nptest.assert_equal(np.squeeze(read_img[0]), self.multich_slice)

    def test_volumeiter_multichannel(self):
        with readers.get_volume_reader(self.multi_path) as reader:
            read_img = list(reader)

        nptest.assert_equal(np.squeeze(read_img[1]), self.multich_slice)

    def test_volumeiter_multichannel_reading_two_channels(self):
        with readers.get_volume_reader(self.multi_path) as reader:
            read_img = list(reader[:, [0, 2]])

        nptest.assert_equal(read_img[1], self.multich_slice[..., [0, 2]])

    def test_read_volume_if_reads_correctly(self):
        stack = read_volume(self.image_path)

        nptest.assert_equal(stack, self.third_channel)

    def test_volread_if_reads_correctly_third_channel(self):
        stack = imageio.volread(self.image_path, channels=2)

        nptest.assert_equal(stack, self.third_channel)

    def test_volread_if_reads_correctly_first_channel(self):
        stack = imageio.volread(self.image_path, channels=0)

        nptest.assert_equal(stack, self.first_channel)

    def test_volread_if_reads_correctly_two_channels(self):
        stack = imageio.volread(self.image_path, channels=[0, 1])

        expected_stack = np.stack((self.first_channel, self.second_channel),
                                  axis=-1)

        nptest.assert_equal(stack, expected_stack)

    def test_volumeiter_if_reads_correctly_third_channel(self):
        with readers.get_volume_reader(self.image_path) as reader:
            stack = reader[:, 2]

            nptest.assert_equal(np.squeeze(stack), self.third_channel)

    def test_volumeiter_if_reads_correctly_two_channels(self):
        with readers.get_volume_reader(self.image_path) as reader:
            stack = reader[:, [0, 1]]
            expected_stack = np.stack((self.first_channel, self.second_channel), axis=-1)

            nptest.assert_equal(np.squeeze(stack), expected_stack)

    def test_volumeiter_from_tiff_can_be_closed(self):
        stack = readers.get_volume_reader(self.image_path)[:, [0, 1]]
        stack.close()
        one_channel = readers.read_one_channel_volume(self.image_path, channels=[0, 2], series=0)
        one_channel.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path)


@pytest.mark.io
class TestVolumeIter(unittest.TestCase):
    def setUp(self):
        self.reader = mock.Mock()
        self.reader.dimensions = {
            'z': 5,
            'x': 10,
            'y': 10,
            'c': 2
        }

        def get_data(z, c):
            return np.array([
                [z, c],
                [z, c]
            ])
        self.get_data = get_data
        self.reader.get_data = self.get_data

    def test_if_reader_is_called_right_with_single_slice(self):
        volume_iter = readers.VolumeIter(self.reader)
        image = volume_iter[0, 1]
        expected_image = np.array([
            [0, 1],
            [0, 1]
        ]).reshape((1, 2, 2, 1))

        nptest.assert_equal(image, expected_image)

    def test_if_reader_is_called_right_with_slice_on_z(self):
        volume_iter = readers.VolumeIter(self.reader)
        image = volume_iter[0:3, 0]
        expected_image = np.array([
            [
                [0, 0],
                [0, 0],
            ],
            [
                [1, 0],
                [1, 0]
            ],
            [
                [2, 0],
                [2, 0]
            ]
        ]).reshape((3, 2, 2, 1))

        nptest.assert_equal(image, expected_image)

    def test_if_reader_is_called_right_with_slice_on_c(self):
        volume_iter = readers.VolumeIter(self.reader)
        image = volume_iter[0, :]
        expected_image = np.array([
            [
                [0, 0],
                [0, 0],
            ],
            [
                [0, 1],
                [0, 1]
            ],
        ])
        expected_image = np.moveaxis(expected_image, 0, -1)[np.newaxis, ...]

        nptest.assert_equal(image, expected_image)
