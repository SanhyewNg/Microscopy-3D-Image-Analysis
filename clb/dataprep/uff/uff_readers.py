import json
import os

import numpy as np
from PIL import Image
from bioformats.omexml import OMEXML

from clb.dataprep.uff.utils import SUPPORTED_DATA_FILE_EXTENSIONS
from clb.dataprep.utils import parse_omexml_metadata


class NotSupportedUFFError(Exception):
    """Raised while reading UFF with unsupported features."""


class UFFReader:
    """
    Class used for reading UFF files.

    UFF structure::

        .
        ├── data
        ├── info.json
        ├── metadata.xml
        ├── thumb.png
        └── thumbs

    Generally in UFF, a tile can be indexed using six 'coordinates':
    x, y, z, m (magnitude), t (time), c (channel).
    For now UFFReader only supports x, y, z, c.
    Folder structure is given by format: data/z{z}/c{c}/x{x}_y{y}.{dataFileExtension}.
    If either z or c is equal to one, we skip it in the path.

    Attributes:
        metadata_xml (bioformats.omexml.OMEXML): OME-XML metadata.
        info_json (dict): Info data about UFF structure.
        dimensions (dict): x,y,z,t,c dimensions extracted from info_json.
    """

    def __init__(self, path):
        """
        Args:
            path: path to UFF.

        Raises:
            NotSupportedUFFError: thrown when dealing with unsupported UFFs
        """
        self._data_path = os.path.join(path, "data")

        with open(os.path.join(path, "metadata.xml"), 'r') as f:
            self.metadata_xml = OMEXML(f.read())
        with open(os.path.join(path, "info.json"), 'r') as f:
            self.info_json = json.load(f)

        if self.info_json['dataFileExtension'] not in SUPPORTED_DATA_FILE_EXTENSIONS:
            raise NotSupportedUFFError("Not supported file extension: {}".format(self.info_json['dataFileExtension']))

        if self._is_tile_incompatible():
            raise NotSupportedUFFError("Not supported tiling")

        self.dimensions = self.info_json['dimensions']
        self.dimensions['x'] = self.info_json['tile']['width']
        self.dimensions['y'] = self.info_json['tile']['height']

        if self.dimensions['t'] > 1:
            raise NotSupportedUFFError("Not supported: t = {} > 1".format(self.dimensions['t']))

    def get_data(self, z, c):
        """Get tile for given z and c indices.

        Args:
            z (int): z index
            c (int): c index (channel)

        Returns:
            np.ndarray: requested tile of shape (x, y)

        Raises:
            IndexError: When z or c are out of range
        """
        if z not in range(self.dimensions['z']):
            raise IndexError("z index out of range")
        if c not in range(self.dimensions['c']):
            raise IndexError("c index out of range")

        tile_path = self._get_tile_path(z, c)
        tile = Image.open(tile_path)

        return np.array(tile)

    def get_metadata(self):
        """Get metadata for UFF.

        Metadata is a dictionary with information extracted from OME-XML file shipped in UFF.

        Keys:
         - Name: str
         - elements of utils.PIXELS_ATTRIBUTES (if present in OME-XML), all str
         - Channels: array of size 'SizeC'. Each element is a dict that might contains some of
         the elements from utils.CHANNEL_ATTRIBUTES

        Returns:
             dict
        """
        image = self.metadata_xml.image(0)
        metadata = parse_omexml_metadata(image)

        return metadata

    def _is_tile_incompatible(self):
        """Checks if there is more than one magnitude level or if there is more than one tile at the lowest level."""
        lod = self.info_json['levelsOfDetail']
        tile = self.info_json['tile']
        return len(lod) != 1 \
            or lod[0]['width'] != tile['width'] \
            or lod[0]['height'] != tile['height']

    def _get_tile_path(self, z, c):
        """Get tile path for given z and c indices"""
        path = ""
        if self.dimensions['z'] > 1:
            path = os.path.join(path, 'z{}'.format(z))
        if self.dimensions['c'] > 1:
            path = os.path.join(path, 'c{}'.format(c))

        return os.path.join(self._data_path, path, "x0_y0.{}".format(self.info_json['dataFileExtension']))
