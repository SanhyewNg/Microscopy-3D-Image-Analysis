"""Module defines additional tools used when reading .lif files.


Attributes:
    default_order (tuple): Expected order of parameters in image name.
    markers_to_channels (dict): Mapping names of markers to channels' numbers.
    steps_to_channels (dict): Mapping number of averaging steps to channels'
                              numbers.
    channels_to_markers (dict): Mapping remainders of dividing channel number
                                by 16 to marker names.
    channels_to_averaging_steps (dict): Mapping results of integer division of
                                        channel number by 16 to number of
                                        averaging steps.
"""
import warnings
from collections import ChainMap

from bioformats.omexml import get_float_attr, get_int_attr

default_order = ('sample', 'size', 'speed', 'region', 'slice')
markers_to_channels = {
    'dapi': {0, 4, 8, 12},
    'pan-cytokeratin': {1, 5, 9, 13},
    'ki67': {2, 6, 10, 14},
    'cd3': {3, 7, 11, 15}
}
steps_to_channels = {
    8: {0, 1, 2, 3},
    4: {4, 5, 6, 7},
    2: {8, 9, 10, 11},
    1: {12, 13, 14, 15},
}
channels_to_markers = {0: 'dapi', 1: 'pan-cytokeratin', 2: 'ki67', 3: 'cd3'}
channels_to_averaging_steps = {0: 8, 1: 4, 2: 2, 3: 1}


class UnrecognizedParametersError(Exception):
    """Raised when order of information in name is unexpected."""


def decode_color(color):
    """Decode color from int.

    Args:
        color (int): Encoded color.

    Returns:
        tuple: Color attribute in form of RGB or None if it's not present.
    """

    # it is in 32-bit RGBA format
    blue = (color >> 8) & 255
    green = (color >> 16) & 255
    red = (color >> 24) & 255

    return red, green, blue


def split_name(name, order=default_order, separator='_'):
    """Read information from name of image.

    Assumes information order is like in `order` and pieces are separated with
    `separator`.

    Args:
        name (str): Name of image.
        order (tuple): Order of information in name.
        separator (str): Separator of information in name.

    Returns:
        dict: Read information.

    Raises:
        UnrecognizedParametersError: If number of pieces in name doesn't match
                             number of pieces in `order`.
    """
    info_parts = name.split(separator)

    if len(info_parts) != len(order):
        raise UnrecognizedParametersError('Unexpected number of'
                                          'information parts: {} != {}.'.
                                          format(len(info_parts), len(order)))

    # Map names from `order` to information from `name`.
    return dict(zip(order, info_parts))


class DenoisingChannelMeta:
    """Channel metadata.

    If `marker` or `averaging_steps` are not given they are inferred in
    following way. `marker` is value at key 'index % 4'
    in `channels_to_markers`, averaging_steps is value at key 'index // 4' in
    `channels_to_averaging_steps`.

    Attributes:
        channel (int): Index of channel in stack.
        marker (str): Name of marker.
        averaging_steps (int): Number of averaging steps.
    """

    def __init__(self, channel=0, marker=None, averaging_steps=None):
        self.channel = channel

        if marker is None:
            marker = channels_to_markers[channel % 4]
        if averaging_steps is None:
            averaging_steps = channels_to_averaging_steps[channel // 4]

        self.marker = marker
        self.averaging_steps = averaging_steps


class DenoisingImageMeta:
    """Image metadata.

    __init__ takes bioformats.OMEXML object as an argument and reads all
    parameters except series from there.

    Attributes:
        series (int): Index of image.
        pixel_type: Type of image (uint8, etc.)
        sample (str): Sample name.
        size (int): Resolution of image.
        speed (int): Speed.
        region (str): Field of view.
        slice (int): Slice (on z axis).
        channels (list): List of channels' metadata.
    """

    def __init__(self, omexml, series=0):
        self.series = series
        self.pixel_type = omexml.Pixels.get_PixelType()

        name_info = split_name(omexml.get_Name())

        # This should set sample, size, speed, region and slice.
        self.sample = name_info['sample']
        self.size = int(name_info['size'])
        self.speed = int(name_info['speed'])
        self.region = name_info['region']
        self.slice = int(name_info['slice'][1:])

        channel_count = omexml.Pixels.get_channel_count()
        self.channels = [DenoisingChannelMeta(index)
                         for index in range(channel_count)]

    def __getitem__(self, key):
        """Return metadata of channel with index `key`.

        Args:
            key (int): Index of channel.

        Returns:
            dict: Metadata of channel. There are following parameters (keys):
                  series
                  pixel_type
                  sample
                  size
                  speed
                  region
                  slice
                  channel
                  marker
                  averaging_steps
        """
        param_dict = self.__dict__.copy()
        param_dict.update(self.channels[key].__dict__)
        param_dict.pop('channels')

        return param_dict


class MarkerMismatchWarning(Warning):
    """Warns that markers inferred from color and index are different."""


class ChannelMeta:
    """Channel metadata.

    Attributes:
        index (int): Channel index.
        color (int): Color attribute (from ome metadata)
        exc_wavelen (float): ExcitationWavelength attribute.
        marker (str): Marker used in this channel, can be one of those in
                      _channels_to_markers.

    Class attributes:
        _channels_to_markers (dict): Mapping channel indices to marker names.
                                     This is specific to our data.
        _colors_to_markers (dict): Mapping colors to marker names. This is
                                   probably also specific to our data. It's
                                   here to check if marker read from index is
                                   the same as read from color.

    Future task: Add exc_wavelen unit.
    """
    _channels_to_markers = {
        0: 'dapi',
        1: 'pan-cytokeratin',
        2: 'ki67',
        3: 'cd3'
    }
    _colors_to_markers = {
        (0, 0, 255): 'dapi',
        (0, 255, 0): 'pan-cytokeratin',
        (255, 255, 0): 'ki67',
        (255, 0, 0): 'cd3'
    }

    def __init__(self, ome_meta, index):
        self.index = index
        self.color = self._get_color(ome_meta)
        self.exc_wavelen = self._get_exc_wavelen(ome_meta)
        self.marker = self._get_marker(self.index, self.color)

    @staticmethod
    def _get_color(ome_meta):
        """Read channel color from channel omexml metadata.

        Args:
            ome_meta (omexml Channel): Metadata.

        Returns:
            tuple: Color attribute in form of RGB or None if it's not present.
        """
        color = get_int_attr(ome_meta.node, 'Color')

        if color is None:
            return None

        return decode_color(color)

    @staticmethod
    def _get_exc_wavelen(ome_meta):
        """Read channel excitation wavelength from omexml metadata.

        Args:
            ome_meta (omexml Channel): Metadata.

        Returns:
            float: Excitation wavelength or None if it's not present.

        """
        exc_wavelen = get_float_attr(ome_meta.node, 'ExcitationWavelength')

        return exc_wavelen

    @classmethod
    def _get_marker(cls, index, color):
        """Read marker from index.

        Function also checks if marker read from color is the same as red from
        index and warns if it isn't.

        Returns:
            str: Marker.
        """
        index_marker = cls._channels_to_markers[index]
        color_marker = cls._colors_to_markers.get(color, None)

        if not color_marker == index_marker:
            msg = MarkerMismatchWarning(
                "Marker read from index doesn't match red from color. "
                "Assuming marker read from index. "
                "(index: {}, color: {})".format(index_marker, color_marker)
            )
            warnings.warn(msg)

        return index_marker

    def get_meta(self):
        """Return channel metadata.

        Returns:
            dict: Metadata, there are following keys:
                  channel_index - index of the channel
                  marker - marker of the channel
                  exc_wavelen - excitation wavelength
        """
        channel_meta = {
            'channel_index': self.index,
            'marker': self.marker,
            'color': self.color,
            'exc_wavelen': self.exc_wavelen
        }

        return channel_meta


class PlaneMeta:
    """Plane metadata.

    Attributes:
        index (int): Plane index.
        channel (int): Channel index.
        z (int): Z coordinate.
    """

    def __init__(self, ome_meta, index):
        self.index = index
        self.channel = self._get_channel(ome_meta)
        self.z = self._get_z(ome_meta)

    @staticmethod
    def _get_channel(ome_meta):
        """Read channel index of plane.

        Args:
            ome_meta (omexml Plane): Metadata.

        Returns:
            int: Channel index.
        """
        channel = ome_meta.get_TheC()

        return channel

    @staticmethod
    def _get_z(ome_meta):
        """Read plane z coordinate.

        Args:
            ome_meta (omexml Plane): Metadata.

        Returns:
            int: Z coordinate.
        """
        z = ome_meta.get_TheZ()

        return z

    def get_meta(self):
        """Return plane metadata.

        Returns:
            dict: Metadata, there are following keys:
                  plane_index - index of the plane
                  z - z coordinate of the plane
        """
        plane_meta = {
            'plane_index': self.index,
            'z': self.z
        }

        return plane_meta


class PixelsMeta:
    """Pixels metadata.

    Attributes:
       size_x (float): Physical x size.
       size_y (float): Physical y size.
       size_z (float): Physical z size.
       size_x_unit (str): Unit of x size.
       size_y_unit (str): Unit of y size.
       size_z_unit (str): Unit of z size.
       type (str): Type of the image (uint8, etc.).
    """

    def __init__(self, ome_meta):
        self.size_x = self._get_size_x(ome_meta)
        self.size_y = self._get_size_y(ome_meta)
        self.size_z = self._get_size_z(ome_meta)
        self.size_x_unit = self._get_size_x_unit(ome_meta)
        self.size_y_unit = self._get_size_y_unit(ome_meta)
        self.size_z_unit = self._get_size_z_unit(ome_meta)
        self.type = self._get_type(ome_meta)
        self.channels = [ChannelMeta(ome_meta.Channel(i), i)
                         for i in range(ome_meta.get_channel_count())]
        self.planes = [PlaneMeta(ome_meta.Plane(i), i)
                       for i in range(ome_meta.get_plane_count())]

    @staticmethod
    def _get_size_x(ome_meta):
        """Read physical x size.

        Args:
            ome_meta (omexml Pixels): Metadata.

        Returns:
            float: Physical x size.
        """
        size_x = ome_meta.get_PhysicalSizeX()

        return size_x

    @staticmethod
    def _get_size_y(ome_meta):
        """Read physical y size.

        Args:
            ome_meta (omexml Pixels): Metadata.

        Returns:
            float: Physical y size.
        """
        size_y = ome_meta.get_PhysicalSizeY()

        return size_y

    @staticmethod
    def _get_size_z(ome_meta):
        """Read physical z size.

        Args:
            ome_meta (omexml Pixels): Metadata.

        Returns:
            float: Physical z size.
        """
        size_z = get_float_attr(ome_meta.node, 'PhysicalSizeZ')

        return size_z

    @staticmethod
    def _get_size_x_unit(ome_meta):
        """Read size x unit.

        Args:
            ome_meta (omexml Pixels): Metadata.

        Returns:
            str: Unit of x size.
        """
        unit = ome_meta.node.get('PhysicalSizeXUnit')

        return unit

    @staticmethod
    def _get_size_y_unit(ome_meta):
        """Read size y unit.

        Args:
            ome_meta (omexml Pixels): Metadata.

        Returns:
            str: Unit of y size.
        """
        unit = ome_meta.node.get('PhysicalSizeYUnit')

        return unit

    @staticmethod
    def _get_size_z_unit(ome_meta):
        """Read size z unit.

        Args:
            ome_meta (omexml Pixels): Metadata.

        Returns:
            str: Unit of z size.
        """
        unit = ome_meta.node.get('PhysicalSizeZUnit')

        return unit

    @staticmethod
    def _get_type(ome_meta):
        """Read image type

        Args:
            ome_meta (omexml Pixels): Metadata.

        Returns:
            str: Image type.
        """
        pixel_type = ome_meta.get_PixelType()

        return pixel_type

    def __getitem__(self, item):
        """Return metadata of plane with given index.

        Args:
            item (int): Plane index.

        Returns:
            ChainMap: Metadata of plane, see ChannelMeta.get_meta,
                      PlaneMeta.get_meta and PixelsMeta.get_meta for list of
                      keys.
        """
        pixels_meta = self.get_meta()
        channel_meta = self.channels[self.planes[item].channel].get_meta()
        plane_meta = self.planes[item].get_meta()
        all_meta = ChainMap(pixels_meta, channel_meta, plane_meta)
        all_meta = dict(all_meta)

        return all_meta

    def get_meta(self):
        """Return metadata of pixels.

        Returns:
            dict: Metadata, there are following keys:
                  size_x - physical x size
                  size_y - physical y size
                  size_z - physical z size
                  size_x_unit - unit of x size
                  size_y_unit - unit of y size
                  size_z_unit - unit of z size
                  type - type of image
        """
        pixels_meta = {
            'size_x': self.size_x,
            'size_y': self.size_y,
            'size_z': self.size_z,
            'size_x_unit': self.size_x_unit,
            'size_y_unit': self.size_y_unit,
            'size_z_unit': self.size_z_unit,
            'type': self.type
        }

        return pixels_meta


class ImageMeta:
    """Metadata of one series.

    Attributes:
        series (int): Series index.
        pixels (PixelsMeta): Pixels metadata.
    """

    def __init__(self, ome_meta, series):
        self.series = series
        self.pixels = PixelsMeta(ome_meta.Pixels)

    def __getitem__(self, item):
        """Return metadata of given plane as mapping.

        Also include information about channel and image as whole.

        Args:
            item (int): Index of plane.

        Returns:
            ChainMap: Metadata of plane, see ImageMeta.get_meta and
                      PixelsMeta.__getitem__ for list of keys.
        """
        image_meta = self.get_meta()
        plane_meta = self.pixels.__getitem__(item)
        all_meta = ChainMap(image_meta, plane_meta)
        all_meta = dict(all_meta)

        return all_meta

    def get_meta(self):
        """Return metadata of image.

        Returns:
            dict: Metadata, there are following keys:
                  series - image series.
        """
        image_meta = {'series': self.series}

        return image_meta
