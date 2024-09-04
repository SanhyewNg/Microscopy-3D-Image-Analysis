import os
import pytest
import unittest
from unittest.mock import call, mock_open, patch

import numpy as np

from clb.dataprep.uff.uff_writers import MetadataError, UFFWriter


@pytest.mark.io
@pytest.mark.os_specific
@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs')
@patch('PIL.Image.Image')
class TestUFFWriter(unittest.TestCase):
    def setUp(self):
        self.path = "/path/to/uff"
        self.regular_dimensions = (3, 128, 128, 2)
        self.metadata = {'Name': '/path/to/file/file.lif',
                         'PhysicalSizeX': '0.5681818181818181',
                         'PhysicalSizeXUnit': 'µm',
                         'PhysicalSizeY': '0.5681818181818181',
                         'PhysicalSizeYUnit': 'µm',
                         'PhysicalSizeZ': '2.0014302564102566',
                         'PhysicalSizeZUnit': 'µm',
                         'SizeC': '2', 'SizeT': '1', 'SizeX': '128',
                         'SizeY': '128', 'SizeZ': '3', 'Type': 'uint8',
                         'Channels': [{'Color': '65535',
                                       'ExcitationWavelengthUnit': 'nm',
                                       'ExcitationWavelength': '561.0'},
                                      {'Color': '-65281',
                                       'ExcitationWavelengthUnit': 'nm',
                                       'ExcitationWavelength': '488.0'}]}

    def test_init_raises_exception_with_wrong_metadata(self, im, mkd, op):
        for i in range(len(self.regular_dimensions)):
            with self.subTest("Wrong dimension: {}".format(i)):
                wrong_dimensions = [*self.regular_dimensions]
                wrong_dimensions[i] = 1
                z, y, x, c = wrong_dimensions
                data = np.arange(z * y * x * c)
                data.shape = (z, y, x, c)
                with self.assertRaises(MetadataError):
                    UFFWriter(self.path, data, self.metadata)

    def test_create_colors_list_returns_list_with_proper_size(self, im, mkd, op):
        size = 42
        color_list = UFFWriter.create_colors_list(size)
        self.assertEqual(size, len(color_list))

    def test_create_colors_list_returns_list_with_colors(self, im, mkd, op):
        size = 2 ** 5
        color_list = UFFWriter.create_colors_list(size)
        color_array = np.asarray(color_list)
        self.assertEqual((size, 4), np.asarray(color_list).shape)

        # Check if RGB values are correct
        self.assertTrue(np.all(np.logical_and(color_array[:, 0:3] >= 0, color_array[:, 0:3] <= 255)))
        # Check if Alpha is 255 (non transparent)
        self.assertTrue(np.all(color_array[:, 3] == 255))

    def test_write_creates_proper_dir_tree_in_regular_case(self, im, mkd, op):
        z, y, x, c = self.regular_dimensions
        data = np.arange(z * y * x * c)
        data.shape = (z, y, x, c)
        calls = [call(self.path)]
        for i in range(z):
            for j in range(c):
                calls.append(call(os.path.join(self.path, "data/z{}/c{}/".format(i, j))))
                calls.append(call(os.path.join(self.path, "thumbs/z{}/c{}/".format(i, j))))
        wr = UFFWriter(self.path, data, self.metadata)
        wr.write()
        mkd.assert_has_calls(calls)

    def test_write_creates_proper_dir_tree_with_z_eq_1(self, im, mkd, op):
        z, y, x, c = self.regular_dimensions
        z = 1
        self.metadata['SizeZ'] = '1'
        data = np.arange(z * y * x * c)
        data.shape = (z, y, x, c)
        calls = [call(self.path)]
        for i in range(c):
            calls.append(call(os.path.join(self.path, "data/c{}/".format(i))))
            calls.append(call(os.path.join(self.path, "thumbs/c{}/".format(i))))
        wr = UFFWriter(self.path, data, self.metadata)
        wr.write()
        mkd.assert_has_calls(calls)

    def test_write_creates_proper_dir_tree_with_c_eq_1(self, im, mkd, op):
        z, y, x, c = self.regular_dimensions
        c = 1
        self.metadata['SizeC'] = '1'
        data = np.arange(z * y * x * c)
        data.shape = (z, y, x, c)
        calls = [call(self.path)]
        for i in range(z):
            calls.append(call(os.path.join(self.path, "data/z{}/".format(i))))
            calls.append(call(os.path.join(self.path, "thumbs/z{}/".format(i))))
        wr = UFFWriter(self.path, data, self.metadata)
        wr.write()
        mkd.assert_has_calls(calls)

    def test_write_creates_info_json_and_metadata_xml_files(self, im, mkd, op):
        z, y, x, c = self.regular_dimensions
        data = np.arange(z * y * x * c)
        data.shape = (z, y, x, c)
        wr = UFFWriter(self.path, data, self.metadata)
        wr.write()
        calls = [call(os.path.join(self.path, "info.json"), 'w'), call(os.path.join(self.path, "metadata.xml"), 'w')]
        op.assert_has_calls(calls, any_order=True)

    def test_write_creates_tile_and_thumbnail_for_every_z_and_c(self, im, mkd, op):
        z, y, x, c = self.regular_dimensions
        data = np.arange(z * y * x * c)
        data.shape = (z, y, x, c)
        wr = UFFWriter(self.path, data, self.metadata)
        wr.write()
        tile_calls = [call()._new()._new().save('/path/to/uff/data/z{}/c{}/x0_y0.png'.format(i, j))
                      for i in range(z) for j in range(c)]
        thumbnails_calls = [call()._new()._new().save('/path/to/uff/thumbs/z{}/c{}/thumb.png'.format(i, j))
                            for i in range(z) for j in range(c)]

        im.assert_has_calls(tile_calls, any_order=True)
        im.assert_has_calls(thumbnails_calls, any_order=True)

    @patch('PIL.Image.fromarray')
    def test_write_saves_correct_rgba_images(self, fr_arr, im, mkd, op):
        z, y, x, c = (1, 3, 2, 1)
        data = np.arange(z * y * x * c)
        data.shape = (z, y, x, c)
        data[0, :, :, 0] = [[8, 0],
                            [8, 0],
                            [5, 0]]

        self.metadata['SizeZ'] = str(z)
        self.metadata['SizeX'] = str(x)
        self.metadata['SizeY'] = str(y)
        self.metadata['SizeC'] = str(c)

        wr = UFFWriter(self.path, data, self.metadata)
        wr.write()

        args, kwargs = fr_arr.call_args_list[0]
        image, mode = args[0], args[1]

        self.assertEqual('RGBA', mode)
        self.assertEqual((y, x, 4), image.shape)

        # Check if 8 was mapped to the same color everywhere
        self.assertTrue(all(i == j for (i, j) in zip(image[0, 0, :], image[1, 0, :])))
        # Check if 8 and 5 were mapped to different colors
        self.assertFalse(all(i == j for (i, j) in zip(image[0, 0, :], image[2, 0, :])))
        # Check if 0 was mapped to transparent color
        self.assertTrue(all(i == j for (i, j) in zip((255, 255, 255, 0), image[0, 1, :])))
        self.assertTrue(all(i == j for (i, j) in zip(image[0, 1, :], image[1, 1, :])))
        self.assertTrue(all(i == j for (i, j) in zip(image[1, 1, :], image[2, 1, :])))

    @patch('PIL.Image.new')
    def test_write_saves_correct_16bits_images(self, im_new, im, mkd, op):
        z, y, x, c = (1, 3, 2, 1)
        data = np.arange(z * y * x * c, dtype=np.uint16)
        data.shape = (z, y, x, c)

        data_slice = np.array([[1, 0], [1, 0], [2 ** 16 - 1, 0]], dtype='uint16')
        data[0, :, :, 0] = data_slice
        data_slice_as_bytes = data_slice.tobytes()

        self.metadata['SizeZ'] = str(z)
        self.metadata['SizeX'] = str(x)
        self.metadata['SizeY'] = str(y)
        self.metadata['SizeC'] = str(c)

        wr = UFFWriter(self.path, data, self.metadata)
        wr.write(color=False)

        args_sequence = zip(im_new.call_args_list, im_new().frombytes.call_args_list)
        for args in args_sequence:
            image_new_args, _ = args[0]
            frombytes_args, _ = args[1]
            self.assertCountEqual(('I', data_slice.T.shape), image_new_args)
            self.assertCountEqual((data_slice_as_bytes, 'raw', 'I;16'), frombytes_args)
