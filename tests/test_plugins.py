import pytest
import unittest
import unittest.mock as mock

import numpy as np

from clb.dataprep.imageioplug.lifformat import LifFormat
from clb.dataprep.imageioplug.tifformat import (ChannelWarning,
                                                 SwappedAxesWarning,
                                                 TiffFormat)


@pytest.mark.io
class TestTiffFormat(unittest.TestCase):
    def setUp(self):
        self.mock_fileobj = mock.Mock()
        self.mock_opener = mock.Mock(return_value=self.mock_fileobj)
        self.mock_swap_checker = mock.Mock()

        self.filename = 'test_filename'
        self.mock_request = mock.MagicMock()
        self.mock_request.mode = mock.MagicMock()
        self.mock_request.__getitem__ = mock.Mock(return_value='i')
        self.mock_request.get_local_filename = mock.Mock(return_value=
                                                         self.filename)
        self.mock_request.kwargs = dict(opener=self.mock_opener,
                                        swap_checker=self.mock_swap_checker)

    def test_if__can_read_returns_true_when_filename_ends_with_tif(self):
        mock_request = mock.Mock()
        mock_request.filename = 'file.tif'
        tiff_format = TiffFormat('TiffFormat', 'It reads tiff.')

        output = tiff_format._can_read(mock_request)

        self.assertTrue(output)

    def test_if__can_read_returns_false_when_filename_not_end_with_tif(self):
        mock_request = mock.Mock()
        mock_request.filename = 'file.lif'
        tiff_format = TiffFormat('TiffFormat', 'It reads tiff.')

        output = tiff_format._can_read(mock_request)

        self.assertFalse(output)

    def test_if__can_write_returns_false(self):
        mock_request = mock.Mock()
        mock_request.filename = 'file.tif'
        tiff_format = TiffFormat('TiffFormat', 'It reads tiff.')

        output = tiff_format._can_write(mock_request)

        self.assertFalse(output)

    def test__open_if_opener_is_called(self):
        TiffFormat.Reader('TiffFormat', self.mock_request)

        self.mock_opener.assert_called_with(self.filename)

    def test__open_if_swap_checker_is_called(self):
        TiffFormat.Reader('TiffFormat', self.mock_request)

        self.mock_swap_checker.assert_called_once_with(self.mock_fileobj)

    def test__open_if_swap_checker_is_not_called_without_flag(self):
        self.mock_request.kwargs['swap_checker'] = None
        TiffFormat.Reader('TiffFormat', self.mock_request)

        self.mock_swap_checker.assert_not_called()

    def test__open_if_warning_is_raised_when_swapping(self):
        with self.assertWarns(SwappedAxesWarning):
            TiffFormat.Reader('TiffFormat', self.mock_request)

    def test__get_length_output_when_axes_swapped(self):
        self.mock_swap_checker.return_value = True
        reader = TiffFormat.Reader('TiffFormat', self.mock_request)

        with mock.patch.object(reader, '_fp') as mock_fp:
            length = 3
            mock_fp.rdr.getSizeT = mock.Mock(return_value=length)

            self.assertEqual(reader._get_length(), length)

    def test__get_length_output_when_axes_not_swapped(self):
        self.mock_swap_checker.return_value = False
        reader = TiffFormat.Reader('TiffFormat', self.mock_request)

        with mock.patch.object(reader, '_fp') as mock_fp:
            length = 3
            mock_fp.rdr.getSizeZ = mock.Mock(return_value=length)

            self.assertEqual(reader._get_length(), length)

    def test__get_data_if_read_is_called_with_right_args_with_swap(self):
        self.mock_swap_checker.return_value = True
        reader = TiffFormat.Reader('TiffFormat', self.mock_request)
        im = np.zeros((3, 3))

        with mock.patch.object(reader._fp, 'read',
                               new=mock.Mock(return_value=im)) as mock_read:
            reader._get_data(0, channels=1)

            mock_read.assert_called_with(t=0, rescale=False, c=1)

    def test__get_data_if_read_is_called_with_right_args_without_swap(self):
        self.mock_swap_checker.return_value = False
        reader = TiffFormat.Reader('TiffFormat', self.mock_request)
        im = np.zeros((3, 3, 2))

        with mock.patch.object(reader._fp, 'read',
                               new=mock.Mock(return_value=im)) as mock_read:
            reader._get_data(0, channels=1)

            mock_read.assert_called_with(z=0, rescale=False, c=1, )

    def test__get_data_if_warning_is_raised(self):
        self.mock_swap_checker.return_value = False
        reader = TiffFormat.Reader('TiffFormat', self.mock_request)
        im = np.zeros((3, 3, 2))

        with mock.patch.object(reader._fp, 'read',
                               new=mock.Mock(return_value=im)):
            with self.assertWarns(ChannelWarning):
                reader._get_data(0, channels=1)


@pytest.mark.io
class TestLifFormat(unittest.TestCase):
    def setUp(self):
        self.mock_fileobj = mock.Mock()
        self.mock_opener = mock.Mock(return_value=self.mock_fileobj)

        self.filename = 'test_filename'
        self.mock_request = mock.Mock()
        self.mock_request.mode = mock.MagicMock()
        self.mock_request.mode.__getitem__ = mock.Mock(return_value='i')
        self.mock_request.get_local_filename = mock.Mock(return_value=
                                                         self.filename)
        self.mock_request.kwargs = dict(opener=self.mock_opener)

    def test_if__can_read_returns_true_when_filename_ends_with_lif(self):
        mock_request = mock.Mock()
        mock_request.filename = 'file.lif'
        lif_format = LifFormat('LifFormat', 'It reads lif.')

        output = lif_format._can_read(mock_request)

        self.assertTrue(output)

    def test_if__can_read_returns_false_when_filename_not_end_with_lif(self):
        mock_request = mock.Mock()
        mock_request.filename = 'file.tif'
        lif_format = LifFormat('LifFormat', 'It reads lif.')

        output = lif_format._can_read(mock_request)

        self.assertFalse(output)

    def test_if__can_write_returns_false(self):
        mock_request = mock.Mock()
        mock_request.filename = 'file.lif'
        lif_format = LifFormat('LifFormat', 'It reads lif.')

        output = lif_format._can_write(mock_request)

        self.assertFalse(output)

    def test__get_data_if_read_is_called_with_right_args(self):
        reader = LifFormat.Reader('LifFormat', self.mock_request)
        mock_lif_reader = mock.Mock()
        mock_lif_reader.get_metadata = mock.Mock(return_value={})

        with mock.patch.object(reader, '_fp', new=mock_lif_reader):
            reader._get_data(0, series=0, channels=1)

            mock_lif_reader.get_data.assert_called_with(z=0, c=1)
