"""Module defines tool for reading images from different formats."""
import collections as colls
import os
import numpy as np

import clb.dataprep.lif.lif_readers as lifreaders
import clb.dataprep.tif.tif_readers as tifreaders
import clb.dataprep.uff.uff_readers as uffreaders
import clb.dataprep.utils as utils


def get_series_nums(path, series):
    """Get list of those series from `series` that can be found in file.

    If file doesn't support series it's assumed to have one series.

    Args:
        path (str): Path to the file.
        series (list|int|None): Desired series. If None all possible series
                                indices will be returned.

    Returns:
        np.ndarray: Desired series that can be found in file.
    """
    try:
        reader = _get_reader(path, series=0)
        series_num = reader.dimensions.get('s', 1)
        all_series = np.arange(series_num)
        chosen_series = all_series[series].reshape(-1)

        if isinstance(chosen_series, np.int64):
            chosen_series = [chosen_series]

        return chosen_series
    finally:
        try:
            reader.close()
        except (UnboundLocalError, AttributeError):
            pass


def get_metadata(path, series):
    """Get metadata of `series` in given file.

    Args:
        path (str): Path to the file.
        series (int): Series to read metadata from.

    Returns:
        dict: Series metadata.
    """
    with get_volume_reader(path, series=series) as reader:
        metadata = reader.metadata

        return metadata


def _get_reader(path, series=0):
    if path.endswith('.tif'):
        reader = tifreaders.TifReader(path)
    elif path.endswith('.lif'):
        reader = lifreaders.LifReader(path, series=series)
    elif os.path.isdir(path):
        reader = uffreaders.UFFReader(path)
    else:
        raise utils.FileFormatError(
            'File format should be either .tif .lif or .uff.')

    return reader


def get_volume_reader(path, series=0):
    """Get reader right for given file, wrapped with VolumeIter.

    File is recognized by extension, if it's extension is not .tif nor .lif
    it should be directory treated as .uff file.

    Args:
        path (str): Path to the file.
        series (int): Series to read.

    Returns:
        VolumeIter: Reader for the file.

    Raises:
        FileFormatError: When file is neither .tif, .lif or directory.
    """
    reader = _get_reader(path, series)

    return VolumeIter(reader)


Shape = colls.namedtuple('Shape', ['z', 'y', 'x', 'c'])


class VolumeIter:
    """Class that provides array-like interface for image readers.

    Attributes:
        metadata (dict): Image metadata.
        _reader: Reader for one type of data.
        _transforms (tuple): Transformations that will be applied to each read
                            slice. Each element should be in form of
                            (func, args, kwargs).
    """
    def __init__(self, reader, z_indices=None, c_indices=None, transforms=()):
        """Limit possible indices to read to `z_indices` and `c_indices`.

        Args:
            z_indices: Any argument that works as numpy index, z indices that
                       will be read from file.
            c_indices: Any argument that works as numpy index, c indices that
                       will be read from file.
        """
        self._reader = reader
        self._transforms = transforms
        self.metadata = self._reader.get_metadata()

        self._z_indices = np.arange(reader.dimensions['z'])[z_indices]
        self._z_indices = self._z_indices.reshape(-1)
        self._c_indices = np.arange(reader.dimensions['c'])[c_indices]
        self._c_indices = self._c_indices.reshape(-1)

        self._shape = Shape(z=len(self.z_indices),
                            y=reader.dimensions['y'],
                            x=reader.dimensions['x'],
                            c=len(self.c_indices))

    def _get_slice(self, z):
        """Get all channels at given `z`.

        Transforms are applied after stacking channels.

        Args:
            z (int): z index of slice.

        Returns:
            np.ndarray: Selected slice.
        """
        channels = np.stack([self._reader.get_data(z=z, c=c)
                             for c in self.c_indices], axis=-1)

        for func, args, kwargs in self._transforms:
            channels = func(channels, *args, **kwargs)

        return channels

    @property
    def voxel_size(self):
        """
        Extract voxel size in the data.
        Returns:
            voxel size as (Z,Y,X) tuple
        """
        try:
            return (float(self.metadata['PhysicalSizeZ']),
                    float(self.metadata['PhysicalSizeY']),
                    float(self.metadata['PhysicalSizeX']))
        except KeyError:
            return None

    @property
    def z_indices(self):
        """z indices that reader "looks at"."""
        return self._z_indices

    @property
    def c_indices(self):
        """c indices that reader "looks at"."""
        return self._c_indices

    @property
    def shape(self):
        """Shape of image."""
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, item):
        """Get image given z and c indices.

        Args:
            item: Any argument that works as numpy index. First axis is z,
                  second is c.

        Returns:
            VolumeIter: Volume cut to given indices.
        """
        if not isinstance(item, tuple):
            item = (item,)

        if len(item) == 1:
            z_indices = self._get_indices(all_indices=self.z_indices,
                                          wanted_indices=item[0])
            c_indices = self.c_indices
        elif len(item) == 2:
            z_indices = self._get_indices(all_indices=self.z_indices,
                                          wanted_indices=item[0])
            c_indices = self._get_indices(all_indices=self.c_indices,
                                          wanted_indices=item[1])
        else:
            raise IndexError('Index should have length 2 maximum.')

        new_volume_iter = self.__class__(reader=self._reader,
                                         z_indices=z_indices,
                                         c_indices=c_indices,
                                         transforms=self._transforms)
        return new_volume_iter

    @staticmethod
    def _get_indices(all_indices, wanted_indices):
        """Get indices from `all_indices`.

        If `all_indices` is [0, 3, 4, 5] and `wanted_indices` is [0, 1, 2],
        then function will return [0, 3, 4].

        Args:
            all_indices (np.ndarray): All possible indices.
            wanted_indices (np.ndarray): Indices to extract.

        Returns:
            np.ndarray: Chosen indices.
        """
        try:
            wanted_indices = all_indices[wanted_indices]

            if isinstance(wanted_indices, np.int64):
                wanted_indices = [wanted_indices]

            return wanted_indices
        except IndexError:
            raise IndexError('Passed wrong indices.')

    def transform(self, func, *args, **kwargs):
        """Create new VolumeIter with added transformation.

        Args:
            func (Callable): Transformation to be added.
            *args: Positional arguments to `func`.
            **kwargs: Keyword arguments to `func`.

        Returns:
            VolumeIter: New object with added transform.
        """
        new_transforms = self._transforms + ((func, args, kwargs),)
        new_volume_iter = self.__class__(self._reader,
                                         z_indices=self.z_indices,
                                         c_indices=self.c_indices,
                                         transforms=new_transforms)

        return new_volume_iter

    def to_numpy(self):
        """Convert VolumeIter to numpy array.

        Returns:
            np.ndarray: Volume, shape (Z, Y, X, C).
        """
        slices = [self._get_slice(z) for z in self.z_indices]
        array = np.stack(slices)

        return array

    def close(self):
        """Close wrapped reader if it's possible."""
        try:
            self._reader.close()
        except AttributeError:
            pass

    def __array__(self):
        return self.to_numpy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __len__(self):
        return self.shape.z

    def __iter__(self):
        """Yield consecutive slices in form of numpy arrays.

        Yields:
            Consecutive slices np.ndarray, shape (Y, X, C).
        """
        for z in self.z_indices:
            yield self._get_slice(z)

    def __eq__(self, other):
        """Compare arrays (like numpy arrays).

        Args:
            other (array_like): Array to compare to.

        Returns:
            np.ndarray: Result of elementwise comparison.
        """
        return np.asarray(self) == np.asarray(other)

    def __ne__(self, other):
        return np.asarray(self) != np.asarray(other)


def read_one_channel_volume(path, channels=None, series=0):
    """Read volume and apply max on its channels if it's multichannel.

    Args:
        path (str): Path to file.
        channels (int/list/None): Channels to read, if None all channels will
                                  be read.
        series (int): Series to read.

    Returns:
        VolumeIter: Volume with one channel.
    """
    reader = get_volume_reader(path=path, series=series)
    volume = reader[:, channels]

    if volume.shape.z == 1:
        raise utils.DimensionError('3D image was expected.')

    if volume.shape.c > 1:
        volume = volume.transform(utils.reduce_to_max_channel)

    return volume
