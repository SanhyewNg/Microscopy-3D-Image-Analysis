"""Module defines tools to read images and their metadata from .lif files."""
import logging
import os
import warnings

import daiquiri

import clb.dataprep.utils as utils
from clb.cropping import CropInfo
from clb.yaml_utils import yaml_file

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

class SwappedAxesWarning(Warning):
    """Raised when reader interprets axes in wrong order."""


class TifReader:
    """Reader of tiff files.

    Attributes:
        reader (bioformats.ImageReader): Reader from bioformats.
        dimensions (dict): Dimensions of image, keys: z, y, x, c.
        swap_t_z (bool): Should axes t and z be swapped (t interpreted as
                         z and vice versa).
    """

    def __init__(self, path):
        """Open connection with file.

        Args:
            path (str): Path to the file.
        """
        self.reader = utils.bioformats_opener(path)
        self._metadata = self.load_meta_from_yaml(yaml_file(path))
        self.dimensions = {
            'z': self.reader.rdr.getSizeZ(),
            'x': self.reader.rdr.getSizeX(),
            'y': self.reader.rdr.getSizeY(),
            'c': self.reader.rdr.getSizeC()
        }

        self.swap_t_z = self.reader.rdr.getSizeT() > 1

        if self.swap_t_z:
            self.dimensions['z'] = self.reader.rdr.getSizeT()
            warnings.warn('Reader sees t and z axis as swapped.',
                          SwappedAxesWarning)
        
        self.emit_single_warn = True


    def get_data(self, z, c):
        """Get image with `z` and `c` indices.

        Shape of returned image is (y, x, c).

        Args:
            z (int): z index.
            c (int): c index.

        Returns:
            np.ndarray: Read image, shape (y, x).
        """
        if self.swap_t_z:
            image = self.reader.read(t=z, c=c, rescale=False)
        else:
            image = self.reader.read(z=z, c=c, rescale=False)

        image = self._ensure_array_byteorder(image)

        # Sometimes all channels are read even when only one is specified.
        if image.ndim > 2:
            image = image[..., c]

        return image

    @staticmethod
    def load_meta_from_yaml(yaml_path):
        metadata = {}
        if os.path.isfile(yaml_path):
            crop_infos = CropInfo.load(yaml_path)
            metadata['PhysicalSizeZ'], \
            metadata['PhysicalSizeY'], \
            metadata['PhysicalSizeX'] = map(str, crop_infos[0].voxel_size)
        return metadata

    def get_metadata(self):
        """Return .tif metadata.

        For now we are not reading metadata from .tif but attempt to load it from yaml file.

        Returns:
            dict: Metadata.
        """
        return self._metadata

    def close(self):
        """Close wrapped reader."""
        self.reader.close()

    def _ensure_array_byteorder(self, array):
        """Fix byteorder - little/big endianness. Set to system specific.
        
        Args:
            array (np.array): Array with certain endianness

        Returns: 
            array (np.array): Array with system specific endianness if applicable
        """ 
        if array.dtype.byteorder not in ("=", "|"):
            if self.emit_single_warn: 
                logger.warn("Byteorder of given array is not system specific. Fixing it...")
                self.emit_single_warn = False
            array = array.astype(array.dtype.name)
        return array
