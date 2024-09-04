import json
import pytest
import unittest
from unittest.mock import patch, mock_open, call

from PIL import Image

from clb.dataprep.uff.uff_readers import UFFReader, NotSupportedUFFError

metadata = '<OME xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 ' \
           'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd" ' \
           'xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" ' \
           'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><Instrument ID="Instrument:2"><Microscope ' \
           'Type="Other"/><Objective Immersion="Other" ID="Objective:2:0" Correction="Other"/><Filter ' \
           'ID="Filter:2:0"/><Filter ID="Filter:2:1"/></Instrument><Image ' \
           'Name="/path/to/file/file.lif" ' \
           'ID="Image:0"><AcquisitionDate>2018-09-05T21:36:12</AcquisitionDate><InstrumentRef ' \
           'ID="Instrument:0"/><ObjectiveSettings ID="Objective:0:0"/><Pixels Type="uint8" TimeIncrementUnit="s" ' \
           'TimeIncrement="1.0" SizeZ="2" SizeY="1024" SizeX="1024" SizeT="1" SizeC="2" SignificantBits="8" ' \
           'PhysicalSizeZUnit="µm" PhysicalSizeZ="2.0014302564102566" PhysicalSizeYUnit="µm" ' \
           'PhysicalSizeY="0.5681818181818181" PhysicalSizeXUnit="µm" PhysicalSizeX="0.5681818181818181" ' \
           'Interleaved="false" ID="Pixels:0" DimensionOrder="XYCZT" BigEndian="false"><Channel SamplesPerPixel="1" ' \
           'ID="Channel:0:0" Color="65535"><LightPath><EmissionFilterRef ' \
           'ID="Filter:0:0"/></LightPath></Channel><Channel SamplesPerPixel="1" ID="Channel:0:1" ' \
           'Color="-65281"><LightPath><EmissionFilterRef ID="Filter:0:1"/></LightPath></Channel><MetadataOnly/><Plane '\
           'TheZ="0" TheT="0" TheC="0" DeltaTUnit="s" DeltaT="0.0"/><Plane TheZ="0" TheT="0" TheC="1" DeltaTUnit="s" ' \
           'DeltaT="0.0"/></Pixels></Image></OME> '

info = '{"dataFileExtension":"png","thumbsFileExtension":"png","tile":{"width":1024,"height":1024},"dimensions":{' \
       '"z":2,"t":1,"c":2},"levelsOfDetail":[{"width":1024,"height":1024}]} '


def change_info(f):
    i = json.loads(info)
    f(i)
    return json.dumps(i)


@pytest.mark.io
@pytest.mark.os_specific
class TestUFFReader(unittest.TestCase):
    def setUp(self):
        self.init_read = [mock_open(read_data=metadata).return_value]

    def _set_init_info(self, init_info):
        self.init_read.append(mock_open(read_data=init_info).return_value)

    @patch('builtins.open', new_callable=mock_open)
    def test_init_reads_info_and_metadata(self, mo):
        self.init_read.append(mock_open(read_data=info).return_value)
        mo.side_effect = self.init_read
        UFFReader("/path/to/uff")
        mo.assert_has_calls([call('/path/to/uff/metadata.xml', 'r'), call('/path/to/uff/info.json', 'r')])

    @patch('builtins.open', new_callable=mock_open)
    def test_init_correctly_sets_dimensions(self, mo):
        self.init_read.append(mock_open(read_data=info).return_value)
        mo.side_effect = self.init_read
        reader = UFFReader("/path/to/uff")
        self.assertDictEqual({'x': 1024, 'y': 1024, 'z': 2, 't': 1, 'c': 2}, reader.dimensions)

    @patch('builtins.open', new_callable=mock_open)
    def test_init_raises_exception_with_wrong_extension(self, mo):
        invalid_extension_info = change_info(lambda x: x.update({'dataFileExtension': 'lif'}))
        self._set_init_info(invalid_extension_info)
        mo.side_effect = self.init_read

        with self.assertRaises(NotSupportedUFFError) as e:
            UFFReader("/")
        self.assertEqual("Not supported file extension: lif", str(e.exception))

    @patch('builtins.open', new_callable=mock_open)
    def test_init_raises_exception_with_many_orders_of_magnitude(self, mo):
        invalid_extension_info = change_info(lambda x: x.update({'levelsOfDetail': [{"width": 1024, "height": 1024},
                                                                                    {"width": 512, "height": 512}]}))
        self._set_init_info(invalid_extension_info)
        mo.side_effect = self.init_read

        with self.assertRaises(NotSupportedUFFError) as e:
            UFFReader("/")
        self.assertEqual("Not supported tiling", str(e.exception))

    @patch('builtins.open', new_callable=mock_open)
    def test_init_raises_exception_with_wrong_tiling(self, mo):
        invalid_extension_info = change_info(lambda x: x.update({'levelsOfDetail': [{"width": 512, "height": 512}]}))
        self._set_init_info(invalid_extension_info)
        mo.side_effect = self.init_read

        with self.assertRaises(NotSupportedUFFError) as e:
            UFFReader("/")
        self.assertEqual("Not supported tiling", str(e.exception))

    @patch('builtins.open', new_callable=mock_open)
    def test_init_raises_exception_with_wrong_t(self, mo):
        invalid_extension_info = change_info(lambda x: x.update({'dimensions': {"z": 2, "t": 2, "c": 2}}))
        self._set_init_info(invalid_extension_info)
        mo.side_effect = self.init_read

        with self.assertRaises(NotSupportedUFFError) as e:
            UFFReader("/")
        self.assertEqual("Not supported: t = 2 > 1", str(e.exception))

    @patch('PIL.Image.open')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_data_raises_exception_with_wrong_z(self, mo, im):
        self.init_read.append(mock_open(read_data=info).return_value)
        mo.side_effect = self.init_read
        im.return_value = Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
        reader = UFFReader("/path/to/uff")
        with self.assertRaises(IndexError) as e:
            reader.get_data(z=100, c=0)
        self.assertEqual("z index out of range", str(e.exception))

    @patch('PIL.Image.open')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_data_raises_exception_with_wrong_c(self, mo, im):
        self.init_read.append(mock_open(read_data=info).return_value)
        mo.side_effect = self.init_read
        im.return_value = Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
        reader = UFFReader("/path/to/uff")
        with self.assertRaises(IndexError) as e:
            reader.get_data(z=0, c=100)
        self.assertEqual("c index out of range", str(e.exception))

    @patch('PIL.Image.open')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_data_reads_proper_tile_in_regular_case(self, mo, im):
        self.init_read.append(mock_open(read_data=info).return_value)
        mo.side_effect = self.init_read
        im.return_value = Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
        reader = UFFReader("/path/to/uff")
        reader.get_data(z=0, c=1)
        im.assert_called_once_with("/path/to/uff/data/z0/c1/x0_y0.png")

    @patch('PIL.Image.open')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_data_reads_proper_tile_with_z_eq_1(self, mo, im):
        invalid_extension_info = change_info(lambda x: x.update({'dimensions': {"z": 1, "t": 1, "c": 2}}))
        self._set_init_info(invalid_extension_info)
        mo.side_effect = self.init_read
        im.return_value = Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
        reader = UFFReader("/path/to/uff")
        reader.get_data(z=0, c=1)
        im.assert_called_once_with("/path/to/uff/data/c1/x0_y0.png")

    @patch('PIL.Image.open')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_data_reads_proper_tile_with_c_eq_1(self, mo, im):
        invalid_extension_info = change_info(lambda x: x.update({'dimensions': {"z": 2, "t": 1, "c": 1}}))
        self._set_init_info(invalid_extension_info)
        mo.side_effect = self.init_read
        im.return_value = Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
        reader = UFFReader("/path/to/uff")
        reader.get_data(z=1, c=0)
        im.assert_called_once_with("/path/to/uff/data/z1/x0_y0.png")

    @patch('builtins.open', new_callable=mock_open)
    def test_get_metadata_properly_extract_info_from_xml(self, mo):
        self.init_read.append(mock_open(read_data=info).return_value)
        mo.side_effect = self.init_read
        reader = UFFReader("/path/to/uff")

        expected_result = {'Name': '/path/to/file/file.lif',
                           'PhysicalSizeX': '0.5681818181818181',
                           'PhysicalSizeXUnit': 'µm',
                           'PhysicalSizeY': '0.5681818181818181',
                           'PhysicalSizeYUnit': 'µm',
                           'PhysicalSizeZ': '2.0014302564102566',
                           'PhysicalSizeZUnit': 'µm',
                           'SizeC': '2', 'SizeT': '1', 'SizeX': '1024',
                           'SizeY': '1024', 'SizeZ': '2', 'Type': 'uint8',
                           'Channels': [{'Color': '65535'}, {'Color': '-65281'}]}

        self.assertDictEqual(expected_result, reader.get_metadata())
