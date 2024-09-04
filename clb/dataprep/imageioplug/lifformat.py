import imageio
import numpy as np

from clb.dataprep.lif.lif_readers import LifReader


class LifFormat(imageio.core.Format):
    def _can_read(self, request):
        """Check if this format can read file in `request`.

        Just checks if filename is ending with '.lif'.

        Args:
            request (imageio.core.Request): Request for reading file.

        Returns:
            bool: True if file can be read, false otherwise.
        """
        return request.filename.endswith('.lif')

    def _can_write(self, request):
        """Check if this format can write file in `request`.

        For now it only returns False.

        Args: request (imageio.core.Request): Request for writing file.

        Returns:
            bool: True if file can be written, false otherwise.
        """
        return False

    class Reader(imageio.core.Format.Reader):
        """Reader of .lif files.

        Attributes:
            series (int): Series from which data will be read by default.
            channels (int or list): Channel(s) from which data will be read by
                                    default.
            _fp (LifReader): Reader of .lif files.
        """
        def _open(self, series=0, channels=None, opener=LifReader, **kwargs):
            """Open connection with file.

            Args:
                series (int): From which series reader should read by default.
                channels (int or list): Channel from which data will be read.
                                        User can read data from other channel,
                                        by passing argument channel to get_data
                                        method, but this channel will be used
                                        when iterating over reader. If None all
                                        channels will be read.
                **kwargs: Additional arguments, used for compatibility with
                          other formats.
            """
            self.series = series
            self.channels = channels

            filename = self.request.get_local_filename()
            self._fp = opener(path=filename, series=series)

        def _close(self):
            """Close connection with file."""
            self._fp.close()

        def get_length(self, **kwargs):
            """get_length()

            Get the number of images in the file. (Note: you can also
            use ``len(reader_object)``.)

            The result can be:
                * 0 for files that only have meta data
                * 1 for singleton images (e.g. in PNG, JPEG, etc.)
                * N for image series
                * inf for streams (series of unknown length)
            """
            return self._get_length(**kwargs)

        def _get_length(self, **kwargs):
            """Get number of pages (z size)."""
            length = self._fp.dimensions['z']

            return length

        def _get_data(self, index, **kwargs):
            """Read .lif page at given `index`.

            Args:
                index (int): Index of page to read.
                **kwargs: Additional arguments, for now it can be:
                          series (int): Series to read.
                          channels (int or list): Channel(s) to read.

            Returns:
                tuple: Image (np.ndarray) of shape (y, x, c), metadata (dict).
            """
            channels = kwargs.get('channels', self.channels)
            if isinstance(channels, int):
                channels = [channels]

            if channels is None:
                channels = list(range(self.size_c()))

            stacks = []
            for channel in channels:
                if self.request.mode[1] == 'v':
                    imgs = (self._fp.get_data(z=z, c=channel)
                            for z in range(self._fp.dimensions['z']))
                else:
                    imgs = [self._fp.get_data(z=index, c=channel)]
                # This is stack with just one channel.
                stack = np.stack(imgs)
                stacks.append(stack)

            all_channels = np.stack(stacks, axis=-1)
            all_channels = np.squeeze(all_channels)

            return all_channels, self.get_meta_data(index, **kwargs)

        def get_meta_data(self, index=None, **kwargs):
            """ get_meta_data(index=None)

            Read meta data from the file. using the image index. If the
            index is omitted or None, return the file's (global) meta data.

            Note that ``get_data`` also provides the meta data for the returned
            image as an atrribute of that image.

            The meta data is a dict, which shape depends on the format.
            E.g. for JPEG, the dict maps group names to subdicts and each
            group is a dict with name-value pairs. The groups represent
            the different metadata formats (EXIF, XMP, etc.).
            """
            self._checkClosed()
            meta = self._get_meta_data(index, **kwargs)
            if not isinstance(meta, dict):
                raise ValueError('Meta data must be a dict, not %r' %
                                 meta.__class__.__name__)
            return meta

        def _get_meta_data(self, index, **kwargs):
            """Read metadata of an image.

            `index` argument is only for compatibility with plugin API, its not
            used.

            Args:
                index (int):
                **kwargs
            """
            metadata = self._fp.get_metadata()

            return metadata

        def size_c(self):
            """Return number of channels."""
            return self._fp.dimensions['c']

        def size_x(self):
            return self._fp.dimensions['x']

        def size_y(self):
            return self._fp.dimensions['y']

        def get_series_num(self):
            """Return number of series."""
            return self._fp.dimensions['s']

        def get_name(self):
            """Return name of series."""
            return self._fp.get_metadata()['Name']
