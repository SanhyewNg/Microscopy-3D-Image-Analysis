import json
from datetime import datetime
from io import StringIO
from xml.etree import ElementTree

from bioformats.omexml import OMEXML, uenc

from clb.dataprep.utils import CHANNEL_ATTRIBUTES, PIXELS_ATTRIBUTES


SUPPORTED_DATA_FILE_EXTENSIONS = ['jpg', 'png', 'tiff']

DEFAULT_XML = '<?xml version="1.0" encoding="UTF-8"?> <OME ' \
              'xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" ' \
              'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' \
              'xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME' \
              '/2016-06 ' \
              'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"> ' \
              '<Image ID="Image:0" Name="default.png"> ' \
              '<AcquisitionDate>%(DEFAULT_NOW)s</AcquisitionDate> <Pixels ' \
              'BigEndian="false" DimensionOrder="XYCZT" ' \
              'ID="Pixels:0" Interleaved="false" SizeC="1" SizeT="1" ' \
              'SizeX="512" SizeY="512" SizeZ="1" Type="uint8"> ' \
              '<Channel ID="Channel:0:0" SamplesPerPixel="1" /> </Pixels> ' \
              '</Image> </OME> '


def build_ome_xml(metadata):
    """

    Args:
        metadata: dictionary with metadata that is needed in OME-XML file.

    Returns:
        str: OME-XML for given data

    data example::

        {'Name': '/path/to/file/file.lif',
         'PhysicalSizeX': '0.5681818181818181',
         'PhysicalSizeXUnit': 'µm',
         'PhysicalSizeY': '0.5681818181818181',
         'PhysicalSizeYUnit': 'µm',
         'PhysicalSizeZ': '2.0014302564102566',
         'PhysicalSizeZUnit': 'µm',
         'SizeC': '2', 'SizeT': '1', 'SizeX': '1024',
         'SizeY': '1024', 'SizeZ': '2', 'Type': 'uint8',
         'Channels': [{'Color': '65535',
                       'ExcitationWavelengthUnit': 'nm',
                       'ExcitationWavelength': '561.0'},
                      {'Color': '-65281',
                       'ExcitationWavelengthUnit': 'nm',
                       'ExcitationWavelength': '488.0'}]}

    """
    o = OMEXML(DEFAULT_XML)

    o.image().set_Name(metadata['Name'])
    o.image().set_AcquisitionDate(datetime.now().strftime("Y-%m-%dT%H:%M:%S"))

    for attr in PIXELS_ATTRIBUTES:
        if attr in metadata:
            o.image().Pixels.node.set(attr, metadata[attr])

    n_channels = len(metadata['Channels'])
    o.image().Pixels.set_channel_count(n_channels)

    for i in range(n_channels):
        for attr in CHANNEL_ATTRIBUTES:
            if attr in metadata['Channels'][i]:
                o.image().Pixels.Channel(i).node.set(attr,
                                                     metadata['Channels'][i][
                                                         attr])
        o.image().Pixels.Channel(i).set_ID("Channel:0:{}".format(i))
        # I needed to remove this to pass info in channel name. Not sure if
        # I can do this.
        # if 'Name' in o.image().Pixels.Channel(i).node.attrib:
        #     del o.image().Pixels.Channel(i).node.attrib['Name']

    result = StringIO()
    ElementTree.register_namespace('',
                                   'http://www.openmicroscopy.org/Schemas/OME/2016-06')
    ElementTree.ElementTree(o.root_node).write(result,
                                               encoding=uenc,
                                               method="xml")
    return result.getvalue()


def build_info_json(metadata):
    x = int(metadata['SizeX'])
    y = int(metadata['SizeY'])
    z = int(metadata['SizeZ'])
    c = int(metadata['SizeC'])

    result = dict()
    result['dataFileExtension'] = 'png'
    result['thumbsFileExtension'] = 'png'

    result['tile'] = dict()
    result['tile']['width'] = x
    result['tile']['height'] = y

    result['dimensions'] = dict()
    result['dimensions']['z'] = z
    result['dimensions']['t'] = 1
    result['dimensions']['c'] = c

    result['levelsOfDetail'] = [dict()]
    result['levelsOfDetail'][0]['width'] = x
    result['levelsOfDetail'][0]['height'] = y

    return json.dumps(result)
