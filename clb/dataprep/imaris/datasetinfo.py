import ntpath as nt
import textwrap

import numpy as np
import yaml

from clb.dataprep.imaris.utils import GetImsPaths, update_attrs, load_attrs


class DataSetInfo:
    def __init__(self, parent_ims, mode, image_metadata=None, time=None):
        """Creates a new timepoint with its subclasses.

        Args:
            parent_ims (obj): A parent file object from h5py module.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
            image_metadata (dict/None): Optional image metadata.
            time (str/None): Optional string containing the time file
                was created at.
        """
        self.channel_list = []
        self.num_channel = 0
        if mode == "x":
            self.ims_group = parent_ims.create_group("DataSetInfo")
        elif mode == "r+":
            self.ims_group = parent_ims["DataSetInfo"]
            get_paths = GetImsPaths(self.ims_group)
            for path in get_paths.path_list:
                if "Channel" in path:
                    self.channel_list += [Channel(self.ims_group,
                                                  mode,
                                                  self.num_channel
                                                  )]
                    self.num_channel += 1

        self.subgroup_dict = {'ImarisDataset': ImarisDataSet(self.ims_group,
                                                             mode),
                              "Imaris": Imaris(self.ims_group,
                                               nt.basename(parent_ims.filename),
                                               mode),
                              "Image": Image(self.ims_group,
                                             image_metadata,
                                             mode),
                              "TimeInfo": TimeInfo(self.ims_group, time,
                                                   mode),
                              "Log": Log(self.ims_group, mode)
                              }

    def add_channel(self, mode, data, color_mode, color_value,
                    channel_name):
        self.channel_list += [Channel(self.ims_group,
                                      mode,
                                      self.num_channel)
                              ]
        self.channel_list[-1].add_channel(data.dtype, np.amax(data), color_mode,
                                          color_value, channel_name)
        self.num_channel += 1
        self.subgroup_dict["Image"].update_noc(data.shape)


class ImarisDataSet:
    def __init__(self, parent_ims, mode):
        """Creates ImarisDataset class object to mirror .ims file structure.
        Args:
            parent_ims (obj): A parent group object from h5py module.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
        """
        if mode == "x":
            self.ims_group = parent_ims.create_group("ImarisDataSet")
            self.attrs = {'Creator': 'Imaris x64',
                          'NumberOfImages': 1,
                          'Version': '9.2'
                          }
            update_attrs(self.ims_group, self.attrs)
        elif mode == "r+":
            self.ims_group = parent_ims["ImarisDataSet"]
            self.attrs = load_attrs(self.ims_group)


class Imaris:
    def __init__(self, parent_ims, filename, mode):
        """Creates Imaris class object to mirror .ims file structure.

        Args:
            parent_ims (obj): A parent group object from h5py module.
            filename (str): The name of a file created/read.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
        """
        if mode == "x":
            self.ims_group = parent_ims.create_group("Imaris")
            self.attrs = {'Filename': filename,
                          'ManufactorString': '',
                          'ImageID': '100001',
                          'ManufactorType': '',
                          'ThumbnailMode': 'ThumbnailMIP',
                          'Version': '9.2'
                          }
            update_attrs(self.ims_group, self.attrs)
        elif mode == "r+":
            self.ims_group = parent_ims["Imaris"]
            self.attrs = load_attrs(self.ims_group)


class Image:
    def __init__(self, parent_ims, image_metadata, mode):
        """Creates Image class object to mirror .ims file structure.

        Args:
            parent_ims (obj): A parent group object from h5py module.
            image_metadata (dict): Optional image metadata.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
        """
        if mode == "x":
            self.ims_group = parent_ims.create_group("Image")
            self.data_shape = None

            try:
                self.attrs = image_metadata
                if not image_metadata['PhysicalSizeXUnit'] == \
                        image_metadata['PhysicalSizeYUnit'] \
                        == image_metadata['PhysicalSizeZUnit']:
                    raise Exception("Units don't match.")

                self.attrs = {
                    'Description': 'Segmentation results',
                    'ExtMax0': image_metadata['PhysicalSizeX'],
                    'ExtMax1': image_metadata['PhysicalSizeY'],
                    'ExtMax2': image_metadata['PhysicalSizeZ'],
                    'ExtMin0': 0,
                    'ExtMin1': 0,
                    'ExtMin2': 0,
                    'Name': image_metadata['marker'],
                    'Noc': 0,
                    'ResampleDimensionX': "true",
                    'ResampleDimensionY': "true",
                    'ResampleDimensionZ': "true",
                    # I hardcoded um for now, because there are some problems
                    # with converting it.
                    'Unit': 'um' #if image_metadata['size_x_unit'] == 'μm' else
                    # image_metadata['size_x_unit']
                }
            except KeyError:
                self.attrs = {'Description': 'Segmentation results',
                              'ExtMax0': 0.227,
                              'ExtMax1': 0.227,
                              'ExtMax2': 0.574,
                              'ExtMin0': 0,
                              'ExtMin1': 0,
                              'ExtMin2': 0,
                              'Name': '(name not specified)',
                              'Noc': 0,
                              'ResampleDimensionX': "true",
                              'ResampleDimensionY': "true",
                              'ResampleDimensionZ': "true",
                              'Unit': 'um',
                              }
            update_attrs(self.ims_group, self.attrs)
        elif mode == "r+":
            self.data_shape = None
            self.ims_group = parent_ims["Image"]
            self.attrs = load_attrs(self.ims_group)
            float_attrs = ['ExtMax0', 'ExtMax1', 'ExtMax2',
                           'ExtMin0', 'ExtMin1', 'ExtMin2']
            int_attrs = ['Noc']

            for attr in float_attrs:
                self.attrs[attr] = float(self.attrs[attr])

            for attr in int_attrs:
                self.attrs[attr] = int(self.attrs[attr])

    def update_noc(self, data_shape):
        if self.data_shape is None:
            self.attrs["Z"] = data_shape[0]
            self.attrs["Y"] = data_shape[1]
            self.attrs["X"] = data_shape[2]
            self.attrs["ExtMax0"] = format(self.attrs["ExtMax0"]
                                           * self.attrs["X"], '.3f')
            self.attrs["ExtMax1"] = format(self.attrs["ExtMax1"]
                                           * self.attrs["Y"], '.3f')
            self.attrs["ExtMax2"] = format(self.attrs["ExtMax2"]
                                           * self.attrs["Z"], '.3f')
            self.data_shape = data_shape
        else:
            assert (np.all(np.equal(self.data_shape, data_shape))), \
                'Channel dims not equal'
        try:
            self.attrs['Noc'] = int(self.attrs['Noc'])
            self.attrs['Noc'] += 1
        except KeyError:
            pass

        update_attrs(self.ims_group, self.attrs)


class TimeInfo:
    def __init__(self, parent_ims, time, mode):
        """Creates TimeInfo class object to mirror .ims file structure.

        Args:
            parent_ims (obj): A parent group object from h5py module.
            time (str): Optional string containing the time file was created at.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
         """
        if mode == "x":
            self.ims_group = parent_ims.create_group("TimeInfo")
            self.attrs = {'DatasetTimePoints': '1',
                          'FileTimePoints': '1',
                          'TimePoint1': time
                          }
            update_attrs(self.ims_group, self.attrs)
        elif mode == "r+":
            self.ims_group = parent_ims["TimeInfo"]
            self.attrs = load_attrs(self.ims_group)


class Log:
    def __init__(self, parent_ims, mode):
        """Creates Log class object to mirror .ims file structure.

        Args:
            parent_ims (obj): A parent group object from h5py module.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
        """
        if mode == 'x':
            self.ims_group = parent_ims.create_group("Log")
            self.attrs = {'Entries': '0'}
            update_attrs(self.ims_group, self.attrs)
        elif mode == "r+":
            self.ims_group = parent_ims["Log"]
            self.attrs = load_attrs(self.ims_group)


class Channel:
    def __init__(self, parent_ims, mode, num_channel):
        """Creates Log class object to mirror .ims file structure.

        Args:
            parent_ims (obj): A parent group object from h5py module.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
            num_channel (int): ID number of the channel.
        """

        self.parent_ims = parent_ims
        self.channel_name = "Channel " + str(num_channel)
        self.previous_channel = "Channel " + str(num_channel - 1)
        self.ims_group = self.parent_ims.require_group(self.channel_name)
        self.max_value = 0
        if mode == 'r+':
            self.attrs = load_attrs(self.ims_group)

    @staticmethod
    def _convert_color(color):
        if color.startswith('#'):
            max_color = 255
            colors = textwrap.wrap(color[1:], 2)
            float_colors = [int(color, base=16) / max_color for color in colors]
            colors_str = '{:.3f} {:.3f} {:.3f}'.format(*float_colors)
        else:
            mapping = {"Red": '1.000 0.000 0.000',
                       "red": '1.000 0.000 0.000',
                       "Green": '0.000 1.000 0.000',
                       "green": '0.000 1.000 0.000',
                       "Blue": '0.000 0.000 1.000',
                       "blue": '0.000 0.000 1.000',
                       "Yellow": '1.000 1.000 0.000',
                       "yellow": '1.000 1.000 0.000'}
            colors_str = mapping[color]

        return colors_str

    def add_channel(self, data_dtype, max_color, color_mode,
                    color_value=None, channel_name=None):
        """Adds another channel to the created/read file.
        Args:
            color_mode (str): Specifying channel color mode. "BaseColor" or
            "TableColor".
            data_dtype (dtype): Dtype of the channel added.
            max_color (int): max color value for Table Color mode.
            color_value (str): Colour of the channel added
            channel_name (dict): Optional channel name.
        """
        self.max_value = np.iinfo(data_dtype).max
        assert color_mode in ['BaseColor', 'TableColor'], 'Invalid color mode.'
        if color_mode == 'BaseColor':
            self.attrs = {'Color': self._convert_color(color_value),
                          'ColorMode': 'BaseColor',
                          'ColorOpacity': '1.000',
                          'ColorRange': '0.000, ' +
                                        str("%.3f" % round(float(self.
                                                                 max_value),
                                                           3)),
                          'Description': '(description not specified)',
                          'GammaCorrection': '1.000',
                          'Name': self.channel_name
                          }
        elif color_mode == 'TableColor':
            with open("clb/dataprep/imaris/colortable.yaml", 'r') \
                    as stream:
                colortable = yaml.load(stream)

            self.attrs = {'ColorMode': 'TableColor',
                          'ColorOpacity': '1.000',
                          'ColorRange': '0.000 '+
                                        str("%.3f" % round(float(
                                                                 max_color),
                                                           3)),
                          'ColorTable': colortable,
                          'ColorTableLength': 256,
                          'Description': '(description not specified)',
                          'GammaCorrection': '1.000',
                          'Name': self.channel_name
                          }
        if channel_name is not None:
            self.attrs['Name'] = channel_name
        update_attrs(self.ims_group, self.attrs)
