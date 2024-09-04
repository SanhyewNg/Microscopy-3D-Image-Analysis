import pytest
import sys
import unittest
from unittest import mock

import bioformats
import numpy as np

from clb.dataprep.lif.lif_readers import DenoisingLifReader, LifReader
from clb.dataprep.lif.utils import DenoisingImageMeta


@pytest.mark.io
class TestDenoisingLifReader(unittest.TestCase):
    def setUp(self):
        self.mock_opener = mock.Mock()
        self.filename = 'test_filename'
        self.reader = DenoisingLifReader(self.filename,
                                         opener=self.mock_opener)
        self.data = '''
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06
        http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image Name="Image0_2048_600_FOV2_z1">
        <Pixels>
        <Channel/>
        <Channel/>
        <Channel/>
        <Channel/>
        </Pixels>
        </Image>
        <Image Name="Image1_1024_500_FOV1_z1">
        <Pixels>
        <Channel/>
        <Channel/>
        <Channel/>
        <Channel/>
        </Pixels>
        </Image>
        </OME>'''
        self.ome = bioformats.OMEXML(self.data)

    def test_check_meta_params_values_output_with_missing_params(self):
        meta = DenoisingImageMeta(self.ome.image(0))

        self.assertFalse(self.reader.check_meta_params_values(meta[0],
                                                              dict(unknown=3)))

    def test_check_meta_params_values_output_with_sample(self):
        meta = DenoisingImageMeta(self.ome.image(0))

        self.assertTrue(
            self.reader.check_meta_params_values(meta[0],
                                                 dict(sample='Image0')))

    def test_if_meta_reader_gives_right_output(self):
        with mock.patch('clb.dataprep.lif.lif_readers.bioformats'
                        '.get_omexml_metadata',
                        new=mock.Mock(return_value=self.data)):
            output = self.reader.meta_reader()
            right_output = (tuple(DenoisingImageMeta(self.ome.image(0)))
                            + tuple(DenoisingImageMeta(self.ome.image(1), 1)))

            self.assertTupleEqual(tuple(output), right_output)

    def test_if_get_matching_meta_gives_right_output(self):
        with mock.patch('clb.dataprep.lif.lif_readers.bioformats'
                        '.get_omexml_metadata',
                        new=mock.Mock(return_value=self.data)):
            output = self.reader.get_matching_meta(dict(sample='Image0'))
            right_output = tuple(DenoisingImageMeta(self.ome.image(0)))

            self.assertTupleEqual(tuple(output), right_output)


@pytest.mark.io
class TestLifReader(unittest.TestCase):
    def setUp(self):
        LifReader.__init__ = mock.Mock(return_value=None)

    def test__ensure_array_byteorder(self):
        lr = LifReader()
        lr.dimensions = {"z":1, "c":1}
        lr.emit_single_warn = True
        lr.swap_t_z = False
        lr._series = None
        lr = self.set_reader_output_to_non_system_specific_endianness(lr)
        image = lr.get_data(0,0)
        assert image.dtype.byteorder == "="

    def set_reader_output_to_non_system_specific_endianness(self, instance):
        opposite_endian = ">" if sys.byteorder == "little" else "<"
        instance._reader = mock.MagicMock()
        instance._reader.read = mock.MagicMock(return_value=np.array([[63,63],[1,9]], dtype=opposite_endian+"i2"))
        return instance
