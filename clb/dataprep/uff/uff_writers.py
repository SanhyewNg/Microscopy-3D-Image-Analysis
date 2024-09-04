import colorsys
import math
import os
from itertools import product
from random import shuffle

import numpy as np
from PIL import Image

from clb.dataprep.uff.utils import (build_info_json, build_ome_xml)

THUMBNAIL_SIZE = (256, 256)


class MetadataError(Exception):
    """Raised when metadata doesn't match data"""


class UFFWriter:
    """
    Class used for writing UFF files.

    UFF structure::

        .
        ├── data
        ├── info.json
        ├── metadata.xml
        ├── thumb.png    <- not created
        └── thumbs

    Generally in UFF, a tile can be indexed using six 'coordinates':
    x, y, z, m (magnitude), t (time), c (channel).
    For now UFFWriter only supports x, y, z, c.
    Folder structure for tiles is given by format: data/z{z}/c{c}/x{x}_y{
    y}.{dataFileExtension}.
    Folder structure for thumbnails is given by format: thumbs/z{z}/c{
    c}/thumb.{dataFileExtension}.
    If either z or c is equal to one, we skip it in the path.
    """

    def __init__(self, path, data, metadata):
        """
        Args:
            path (str): place where UFF file will be created
            data (np.ndarray): array of shape (z, y, x, c)
            metadata: (dict): information needed to build OME-XML and info.json

        Metadata keys:
         - Name: str
         - elements of utils.PIXELS_ATTRIBUTES , all str
         - Channels: array of size 'SizeC'. Each element is a dict that
         might contains some of
         the elements from utils.CHANNEL_ATTRIBUTES

        Raises:
            MetadataError: When metadata doesn't match data.
        """
        self._path = path
        self._data = data
        self._metadata = metadata

        self._sizeZ = int(metadata['SizeZ'])
        self._sizeC = int(metadata['SizeC'])
        self._sizeX = int(metadata['SizeX'])
        self._sizeY = int(metadata['SizeY'])

        metadata_shape = (self._sizeZ, self._sizeY, self._sizeX, self._sizeC)

        if metadata_shape != data.shape:
            raise MetadataError(
                "Data and metadata dimensions are not the same! {} != {}".
                    format(data.shape, metadata_shape))

        self.colors_palette = self._create_color_palette()

    def write(self, color=True, colors_palettes=None):
        """
        Function that creates UFF folder structure and write tiles and
        thumbnails.
        Args:
            color (bool): Indicate whether color the output or not
            colors_palettes (Iterable): If color is True, this can be used
                                        to override random colors on each
                                        channel. If it's None random colors
                                        will be used on that channel.
        """
        os.makedirs(self._path)

        info_json = build_info_json(self._metadata)
        with open(os.path.join(self._path, 'info.json'), 'w') as f:
            f.write(info_json)

        ome_xml = build_ome_xml(self._metadata)
        with open(os.path.join(self._path, 'metadata.xml'), 'w') as f:
            f.write(ome_xml)

        for z, z_slice in enumerate(self._data):
            for c in range(z_slice.shape[-1]):
                tile_dir = self._get_dirs_path(z, c, 'data')
                os.makedirs(tile_dir)
                tile_path = os.path.join(tile_dir, 'x0_y0.png')

                thumb_dir = self._get_dirs_path(z, c, 'thumbs')
                os.makedirs(thumb_dir)
                thumb_path = os.path.join(thumb_dir, 'thumb.png')

                image = z_slice[:, :, c]

                if color:
                    if (colors_palettes is not None
                            and colors_palettes[c] is not None):
                        image = colors_palettes[c][image]
                    else:
                        image = self.colors_palette[image]
                    image = image.astype('uint8')

                    tile = Image.fromarray(image, 'RGBA')
                    thumb = Image.fromarray(image, 'RGBA')
                else:
                    assert image.dtype == np.uint8 or image.dtype == np.uint16, \
                        "Float or uint32 array passed to save as non-coloured uff. " \
                        "It will be shrink into uint16 so it may be impossible to reconstruct the original information."

                    image = image.astype('uint16')
                    arr_buf = image.tobytes()

                    tile = Image.new('I', image.T.shape)
                    tile.frombytes(arr_buf, 'raw', 'I;16')

                    thumb = Image.new('I', image.T.shape)
                    thumb.frombytes(arr_buf, 'raw', 'I;16')

                tile.save(tile_path)

                thumb.thumbnail(THUMBNAIL_SIZE)
                thumb.save(thumb_path)

    @staticmethod
    def create_colors_list(size):
        """
        Create list with unique RGBA colors.

        Args:
            size (int): size of created list

        Returns:
            list

        """
        color_types = [(x[0], x[1]) for x in
                       product([0.6, 0.8, 1.0], repeat=2)]
        colors_per_type = math.ceil(size / len(color_types))

        colors = []
        for t in color_types:
            hsv_colors = [(x * 1.0 / colors_per_type, t[0], t[1]) for x in
                          range(colors_per_type)]
            for rgb in hsv_colors:
                rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
                colors.append(tuple(rgb))

        shuffle(colors)
        colors = colors[:size]
        colors = [(c[0], c[1], c[2], 255) for c in colors]

        return colors

    def _create_color_palette(self):
        unique_classes = np.unique(self._data)
        number_of_classes = unique_classes.shape[0]

        colors_list = UFFWriter.create_colors_list(number_of_classes)
        colors_list[0] = (255, 255, 255, 0)  # 0 should be transparent

        max_cell_number = unique_classes[-1]
        colors_palette = np.arange((max_cell_number + 1) * 4)
        colors_palette.shape = (max_cell_number + 1, 4)

        for (u, c) in zip(unique_classes, colors_list):
            colors_palette[u] = c

        return colors_palette

    def _get_dirs_path(self, z, c, base_dir):
        """Get directory for a tile for given z and c indices"""
        relative_path = ""
        if self._sizeZ > 1:
            relative_path = os.path.join(relative_path, 'z{}'.format(z))
        if self._sizeC > 1:
            relative_path = os.path.join(relative_path, 'c{}'.format(c))

        return os.path.join(self._path, base_dir, relative_path, '')
