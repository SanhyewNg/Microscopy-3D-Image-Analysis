"""Module defines tools to read images and their metadata from .lif files."""
import logging
import warnings

import bioformats
import daiquiri
import javabridge

from clb.dataprep.lif.meta_readers import MetaReader
from clb.dataprep.lif.utils import (DenoisingImageMeta,
                                    UnrecognizedParametersError)
from clb.dataprep.utils import bioformats_opener, parse_omexml_metadata

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


class LifReader:
    """Reader for .lif files.

    Attributes:
        dimensions (dict): Dimensions of image, keys: s, z, y, x, c,
                           s is for series.
        _metadata (dict): Metadata of the series.
        _reader (bioformats.ImageReader): Wrapped bioformats reader.
        _series (int): Series to read from.
    """
    def __init__(self, path, series):
        assert series is not None, "For .lif input file, series parameter is required."

        self._reader = bioformats_opener(path)
        self._series = series

        omexml = bioformats.get_omexml_metadata(path)
        parsed_omexml = bioformats.OMEXML(omexml)
        series_metadata = parsed_omexml.image(index=series)
        self._metadata = parse_omexml_metadata(series_metadata)
        self.dimensions = {
            's': parsed_omexml.image_count,
            'z': int(self._metadata['SizeZ']),
            'y': int(self._metadata['SizeY']),
            'x': int(self._metadata['SizeX']),
            'c': int(self._metadata['SizeC'])
        }
        self.emit_single_warn = True

    def get_data(self, z, c):
        """Get image with `z` and `c` indices.

        Args:
            z (int): z index of an image.
            c (int): c (channel) index of an image.

        Returns:
            np.ndarray: Read image, shape (Y, X)

        Raises:
            IndexError: When `z` or `c` indices are wrong.
        """
        if z not in range(self.dimensions['z']):
            raise IndexError('Wrong z index - {}/{}.'
                             .format(z, self.dimensions['z']))
        if c not in range(self.dimensions['c']):
            raise IndexError('Wrong c index - {}/{}.'
                             .format(c, self.dimensions['c']))

        image = self._reader.read(series=self._series, z=z, c=c, rescale=False)

        image = self._ensure_array_byteorder(image)

        return image

    def get_metadata(self):
        """Return series metadata.

        Returns:
            dict: Metadata.
        """
        return self._metadata

    def close(self):
        """Close wrapped reader."""
        self._reader.close()

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tranceback):
        """Close wrapped reader."""
        self.close()


class MetadataError(Exception):
    """Raised when bioformats can't read metadata."""


class DenoisingLifReader(MetaReader):
    """MetaReader customized to read .lif files.


    Attributes:
        reader (bioformats.ImageReader): Data reader.
    """
    def __init__(self, path, opener=bioformats.ImageReader):
        """Initialize reader.

        Args:
            path (str): Path to file.
            opener (Callable): Used to open connection with file. Will be
                               called with filename.
        """
        super().__init__(path)
        self.reader = opener(path)

    def close(self):
        """Close connection with file and schedule closing of vm.
        """
        super().close()

        self.reader.close()

    def meta_reader(self):
        """Read metadata of all images from file.

        Yields:
            dict: Image metadata, see DenoisingImageMeta.__getitem__ for more
                  info.

        Raises:
            MetadataError: When file metadata can't be read.

        Future task: Add test to check warning.
        Future task: Add test to check file type exception.
        """
        try:
            xml = bioformats.get_omexml_metadata(self.path)
            ome_meta = bioformats.OMEXML(xml)

            images_indices = range(ome_meta.get_image_count())

            for index in images_indices:
                try:
                    yield from DenoisingImageMeta(ome_meta.image(index), index)
                except UnrecognizedParametersError:
                    warnings.warn("Couldn't recognize parameters at index {} "
                                  "in file {}".format(index, self.path))

        except javabridge.jutil.JavaException:
            raise MetadataError('Wrong file type.')

    @staticmethod
    def check_meta_params_values(meta, params):
        """Check if parameters in `params` match those in `meta`.

        Currently supported parameters:
        sample (str)
        size (int)
        speed (int)
        region (str)
        slice (int)
        marker (str)
        averaging_steps (int)

        Args:
            meta (dict): Metadata like described in
                         DenoisingImageMeta.__getitem__.
            params: Parameters to check for match.

        Returns:
            bool: True if there is a match, False otherwise.
        """
        try:
            return all(meta[name] == value for name, value in params.items())

        # Catching KeyError in case of unknown parameter.
        except KeyError:
            return False

    def data_reader(self, meta):
        """Read data from given file.

        Args:
            meta (dict): Metadata of image to read.

        Returns:
            np.ndarray: Read image.
        """
        image = self.reader.read(series=meta['series'],
                                 c=meta['channel'],
                                 rescale=False)
        return image
