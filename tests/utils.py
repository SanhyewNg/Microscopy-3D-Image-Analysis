import random
import unittest

import imageio
import numpy as np
import numpy.testing as nptest
import pandas as pd

import clb.dataprep.imaris.ims_file as imsfile


class TestCase(unittest.TestCase):
    def assert_ims_equal_to_tifs(self, ims_path, *tif_paths, slices=None):
        with imsfile.ImsFile(ims_path, 'r+') as file:
            file_content = imsfile.extract_channels(file, channels=-1)
            channels = (self.read_channel(channel) for channel in tif_paths)
            expected_content = np.stack(channels, axis=-1)
            nptest.assert_equal(file_content[slices], expected_content[slices])

    @staticmethod
    def read_channel(channel):
        if isinstance(channel, str):
            image = imageio.volread(channel)
        else:
            path, channel_num = channel
            image = imageio.volread(path)[..., channel_num]

        return image

    def assert_called_once_including(self, mock, **kwargs):
        self.assertEqual(1, len(mock.call_args_list), "Single call expected.")
        dict_args = mock.call_args[1].items()
        for kv in kwargs.items():
            self.assertIn(kv, dict_args)

    @staticmethod
    def assert_tif_equal_to_tif(tif_path, tif_with_expected, slices=None, atol=0):
        file_content = imageio.volread(tif_path)
        expected_content = imageio.volread(tif_with_expected)
        nptest.assert_allclose(file_content[slices], expected_content[slices], atol=atol)

    @staticmethod
    def assert_csv_equal_to_csv(csv_path1, csv_path2):
        csv1_df = pd.read_csv(csv_path1)
        csv2_df = pd.read_csv(csv_path2)

        pd.testing.assert_frame_equal(csv1_df, csv2_df, check_dtype=False)

    @staticmethod
    def set_stop_mock(mock):
        def raise_exception(*args, **kwargs):
            raise MockHitException(str(mock))

        mock.side_effect = raise_exception


class MockHitException(Exception):
    def __init__(self, name):
        super().__init__("Mock was hit: " + name)


def get_random_classes(shape):
    size = shape[-1]
    classes = np.zeros(shape, dtype=np.uint8)
    for id in range(5):
        add_random_spot(classes, id % 2 + 1, size=size // 5)
    return classes


def get_random_labels(shape, cells_num, size=3):
    labels = np.zeros(shape, dtype=np.uint8)
    for cell_id in range(1, cells_num + 1):
        add_random_spot(labels, cell_id, size=size)
    return labels


def add_random_spot(volume, value, size=3):
    slices = []
    for s in volume.shape:
        start = random.randint(0, max(0, s - size))
        end = min(s, start + size)
        slices.append(slice(start, end))
    volume[slices] = value
