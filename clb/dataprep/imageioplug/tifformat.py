import warnings

import bioformats
import imageio
import numpy as np

from clb.dataprep.utils import bioformats_opener


class SwappedAxesWarning(Warning):
    """Raised when reader interprets axes in wrong order."""


class ChannelWarning(Warning):
    """Raised when reader cannot read one channel and reads all of them."""


def _check_if_swap(fileobj):
    """Check if t axis should be interpreted as z axis.

    Function assumes that only size t or only size z should be bigger
    than 1, not both.

    Args:
        fileobj (bioformats.ImageReader): Opened reader.

    Returns:
        bool: True if t size is bigger than 1, False otherwise.

    Raises:
        AssertionError: When t size and z size are both bigger than 1.
    """
    size_z = fileobj.rdr.getSizeZ()
    size_t = fileobj.rdr.getSizeT()
    assert size_t == 1 or size_z == 1, \
        'There is time and depth in data.'

    # We assume in our data there is no time, so if t size is
    # bigger than 1, we recognize t as z.
    return size_t > 1


class TiffFormat(imageio.core.Format):
    def _can_read(self, request):
        """Check if this format can read file in `request`.

        Just checks if filename is ending with '.tif'.

        Args:
            request (imageio.core.Request): Request for reading file.

        Returns:
            bool: True if file can be read, false otherwise.
        """
        return request.filename.endswith('.tif')

    def _can_write(self, request):
        """Check if this format can write file in `request`.

        For now it only returns False.

        Args: request (imageio.core.Request): Request for writing file.

        Returns:
            bool: True if file can be written, false otherwise.
        """
        return False

    class Reader(imageio.core.Format.Reader):
        """Reader of tiff files.

        Attributes:
            channels (int or list): Channel(s) from which data will be read by
                                    default.
            _fp (bioformats.ImageReader): Reader from bioformats.
            swap_t_z (bool): Should axes t and z be swapped (t interpreted as
                             z and vice versa).
        """
        def _open(self, channels=None, opener=bioformats_opener,
                  swap_checker=_check_if_swap,
                  **kwargs):
            """Open connection with file.

            Args:
                channels (int or list): Channel(s) from which data will be
                                        read. User can read data from other
                                        channel, by passing argument channel
                                        to get_data method, but this channel
                                        will be used when iterating over
                                        reader. If None all channels will be
                                        read.
                opener (Callable): Used to open connection with file. It will
                                   be called with filename.
                swap_checker (Callable): Used to check if t and z axis should
                                         be swapped.
                **kwargs: Additional arguments, used for compatibility with
                          other formats.
            """
            self.channels = channels
            filename = self.request.get_local_filename()
            self._fp = opener(filename)

            if swap_checker is not None:
                self.swap_t_z = swap_checker(self._fp)
            else:
                self.swap_t_z = False

            if self.swap_t_z:
                warnings.warn('Reader sees t and z axis as swapped.',
                              SwappedAxesWarning)

        def _close(self):
            """Close connection with file."""
            self._fp.close()

        def _get_length(self):
            """Get number of pages (z size)."""
            # Checking if t axis should be interpreted as z (because of wrong
            # reader interpretation).
            if self.swap_t_z:
                length = self.size_t()
            else:
                length = self.size_z()

            return length

        def _get_data(self, index, **kwargs):
            """Read .tif page at given `index`.

            Shape of returned image is (y, x, c).

            Args:
                index (int): Index of page to read.
                **kwargs: Additional arguments, for now it can be:
                          channels (int or list): Channel(s) to read. If None
                                                  all channels are read.

            Returns:
                tuple: Image (np.ndarray), metadata (dict).
            """
            channels = kwargs.get('channels', self.channels)

            if isinstance(channels, int):
                channels = [channels]

            if channels is None:
                channels = list(range(self.size_c()))

            if self.request.mode[1] == 'v':
                slices = (self._read_one_slice(idx, channels)
                          for idx in range(self.depth_size()))
                im = np.stack(slices)
            else:
                im = self._read_one_slice(index, channels)

            im = np.squeeze(im)

            return im, {}

        def get_meta_data(self, index=None, **kwargs):
            return self._get_meta_data(index, **kwargs)

        def _get_meta_data(self, index, **kwargs):
            """Read metadata of image."""
            metadata = {'shape': (self.size_y(), self.size_x())}

            return metadata

        def _read_one_slice(self, index, channels):
            if self.swap_t_z:
                im = [self._fp.read(c=c, t=index, rescale=False)
                      for c in channels]
            else:
                im = [self._fp.read(c=c, z=index, rescale=False)
                      for c in channels]

            if im[0].ndim > 2:
                warnings.warn("Reader couldn't read just one channel, so it"
                              " read all of them.", ChannelWarning)
                im = im[0][..., channels]
            else:
                im = np.stack(im, axis=-1)

            return im

        def depth_size(self):
            if self.swap_t_z:
                return self.size_t()
            else:
                return self.size_z()

        def size_z(self):
            return self._fp.rdr.getSizeZ()

        def size_c(self):
            return self._fp.rdr.getSizeC()

        def size_t(self):
            return self._fp.rdr.getSizeT()

        def size_x(self):
            return self._fp.rdr.getSizeX()

        def size_y(self):
            return self._fp.rdr.getSizeY()

