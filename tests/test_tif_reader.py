import pytest
import sys
import unittest
from unittest import mock

import numpy as np

from clb.dataprep.tif.tif_readers import TifReader


@pytest.mark.io
class TestTifReader(unittest.TestCase):
    def setUp(self):
        TifReader.__init__ = mock.Mock(return_value=None)

    def test__ensure_array_byteorder(self):
        tr = TifReader()
        tr.emit_single_warn = True
        tr.swap_t_z = False
        tr = self.set_reader_output_to_non_system_specific_endianness(tr)
        image = tr.get_data(0,0)
        assert image.dtype.byteorder == "="

    def set_reader_output_to_non_system_specific_endianness(self, instance):
        opposite_endian = ">" if sys.byteorder == "little" else "<"
        instance.reader = mock.MagicMock()
        instance.reader.read = mock.MagicMock(return_value=np.array([[63,63],[1,9]], dtype=opposite_endian+"i2"))
        return instance
