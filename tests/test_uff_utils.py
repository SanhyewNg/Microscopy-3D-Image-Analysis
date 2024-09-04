import json
import pytest
import unittest
import xml.etree.ElementTree as ElementTree

from clb.dataprep.uff.utils import (CHANNEL_ATTRIBUTES, PIXELS_ATTRIBUTES,
                                    build_info_json, build_ome_xml)


data = {'Name': '/path/to/file/file.lif',
        'PhysicalSizeX': '0.5681818181818181',
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': '0.5681818181818181',
        'PhysicalSizeYUnit': 'µm',
        'PhysicalSizeZ': '2.0014302564102566',
        'PhysicalSizeZUnit': 'µm',
        'SizeC': '2', 'SizeT': '1', 'SizeX': '128',
        'SizeY': '128', 'SizeZ': '3', 'Type': 'uint8',
        'Channels': [{'Name': 'Channel1',
                      'Color': '65535',
                      'ExcitationWavelengthUnit': 'nm',
                      'ExcitationWavelength': '561.0'},
                     {'Name': 'Channel2',
                      'Color': '-65281',
                      'ExcitationWavelengthUnit': 'nm',
                      'ExcitationWavelength': '488.0'}]}

reference_info_json_string = '{"dataFileExtension": "png", "thumbsFileExtension": "png", "tile": {"width": 128, ' \
                             '"height": 128}, "dimensions": {"z": 3, "t": 1, "c": 2}, "levelsOfDetail": [{' \
                             '"width":128, "height": 128}]} '


@pytest.mark.io
class TestUFFUtils(unittest.TestCase):

    def test_build_ome_xml_correctly_builds_xmls(self):
        xml = build_ome_xml(data)
        tree = ElementTree.fromstring(xml)
        ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

        self.assertEqual(data['Name'], tree.find('./ome:Image', ns).attrib['Name'])

        for attr in PIXELS_ATTRIBUTES:
            self.assertEqual(data[attr], tree.find('./ome:Image/ome:Pixels', ns).attrib[attr])

        n_channels = len(data['Channels'])
        channels_attribs = [c.attrib for c in tree.findall('./ome:Image/ome:Pixels/ome:Channel', ns)]

        self.assertEqual(n_channels, len(channels_attribs))

        for i in range(n_channels):
            for attr in CHANNEL_ATTRIBUTES:
                self.assertEqual(data['Channels'][i][attr], channels_attribs[i][attr])

    def test_build_info_json_correctly_builds_json(self):
        computed_info_json = json.loads(build_info_json(data))
        reference_info_json = json.loads(reference_info_json_string)

        self.assertDictEqual(computed_info_json, reference_info_json)
