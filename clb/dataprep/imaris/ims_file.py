import datetime
import os

import h5py
import imageio
import numpy as np

from clb.dataprep.imaris.dataset import DataSet
from clb.dataprep.imaris.datasetinfo import DataSetInfo
from clb.dataprep.imaris.dstimes_and_scene import DatasetTimes, Scene
from clb.dataprep.imaris.utils import update_attrs, load_attrs
import clb.dataprep.readers as readers
from clb.dataprep.utils import save_to_tif

class ImsFile:
    def __init__(self, filepath, mode, image_metadata=None):
        """Object of this class handles an imaris file specified in the
        filepath, either creating it or reading it from the path provided. With
        its subclasses and attributes it mirrors ims file structure allowing
        channel addition and attribute modification.

        Args:
            filepath (path): Path to the file.

            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.

            image_metadata (None, dict): If ims is created from scratch, user
                may add their own metadata in a form of a dict. Default is None.
        """
        if image_metadata is None:
            image_metadata = {}
        valid_mode = {"x", "r+"}
        if mode not in valid_mode:
            raise ValueError("Invalid mode, needs to be either x or r+")
        self.mode = mode
        if self.mode == "x":
            self._create_ims(filepath, image_metadata)
        elif self.mode == "r+":
            self._load_ims(filepath)

    def _create_ims(self, filepath, image_metadata):
        """Creates IMS file from scratch.

        Args:
            filepath (path): Path to the file.
            image_metadata (None, dict): If ims is created from scratch, user
                may add their own metadata in a form of a dict. Default is None.
        """
        try:
            self.imsgroup = h5py.File(filepath, self.mode)
        except OSError:
            os.remove(filepath)
            self.imsgroup = h5py.File(filepath, self.mode)

        self.resolution = 1
        self.attrs = {'DataSetDirectoryName': "DataSet",
                      'DataSetInfoDirectoryName': 'DataSetInfo',
                      'ImarisDataSet': 'ImarisDataSet',
                      'ImarisVersion': '5.5.0',
                      'NumberOfDataSets': 1,
                      'ThumbnailDirectoryName': 'Thumbnail',
                      }
        update_attrs(self.imsgroup, self.attrs)
        now = datetime.datetime.now()
        self.time = now.strftime("%Y-%m-%d %H:%M:%S:%M:%f")

        self.subgroup_list = {"DataSet": DataSet(self.imsgroup, self.mode,
                                                 self.resolution),
                              "DataSetInfo": DataSetInfo(self.imsgroup,
                                                         self.mode,
                                                         image_metadata,
                                                         self.time),
                              "DataSetTimes": DatasetTimes(self.imsgroup,
                                                           self.mode,
                                                           self.time),
                              "Scene": Scene(self.imsgroup, self.mode)
                              }

    def _load_ims(self, filepath):
        """Reads an already existing imaris file and mirrors its data.

        Args:
            filepath (path): Path to the file.
        """
        self.imsgroup = h5py.File(filepath, self.mode)
        self.attrs = load_attrs(self.imsgroup)

        self.subgroup_list = {"DataSet": DataSet(self.imsgroup, self.mode),
                              "DataSetInfo": DataSetInfo(self.imsgroup,
                                                         self.mode,
                                                         image_metadata=None,
                                                         time=None),
                              "DataSetTimes": DatasetTimes(self.imsgroup,
                                                           self.mode),
                              "Scene": Scene(self.imsgroup, self.mode)
                              }

    def add_channel(self, data, color_mode='BaseColor', color_value=None,
                    channel_name=None):
        """Adds another channel to the .ims file, filling in the data and attrs.
        Args:
            data (arr): numpy array of shape (Z, Y, X) containing
                a 3D image
            color_mode (str): Specifying whether channel is in BaseColor
            or TableColor mode.
            color_value (str/None): Color of the channel in BaseColor
            channel_name (str / None): Optional channel name.
        """
        self.subgroup_list["DataSet"].add_channel(data)
        self.subgroup_list["DataSetInfo"].add_channel(self.mode,
                                                      data,
                                                      color_mode,
                                                      color_value,
                                                      channel_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Close corresponding h5py file."""
        self.imsgroup.close()


def extract_channels(ims_file, channels):
    """Extract channels from .ims file.

    Args:
        ims_file (ImsFile): ImsFile object to extract channels from.
        channels (list/int): Channel(s) to extract. Use -1 to extract all
                             channels.

    Returns:
        np.ndarray: Extracted channels, shape (Z, Y, X, C).
    """
    if isinstance(channels, int):
        channels = [channels]

    res_level = ims_file.subgroup_list['DataSet'].resolution_list[0]
    timepoint = res_level.timepoint
    channel_list = timepoint.channel_list

    # All channels should be read if channels is -1.
    if channels == [-1]:
        channels = range(len(channel_list))

    channels_data = (channel_list[i].ims_group['Data'] for i in channels)
    data = np.stack(channels_data, axis=-1)

    return data


def extract_channels_to_tif(ims_path, channels, tif_path):
    """Extract channels from ims file and save them to tif file.

    Args:
        ims_path (str): Path to ims file to extract channels from.
        channels (list/int): Channels to extract.
        tif_path (str): Path to tif file to save to.
    """
    with ImsFile(ims_path, mode='r+') as ims_file:
        data = extract_channels(ims_file, channels)
        save_to_tif(tif_path, data)


def add_channel_from_tif(ims_path, tif_path, color, name, channel=0):
    """Read channel from tif file and add it to ims file.

    Args:
        ims_path (str): Path to ims file.
        tif_path (str): Path to tif file.
        color (str): Channel color. If 'Segmentation' then TableColor mode is used.
        name (str): Channel name.
        channel (int): Index of channel to read from tif file.
    """
    mode = 'r+' if os.path.isfile(ims_path) else 'x'

    with ImsFile(ims_path, mode=mode) as ims_file:
        volume = imageio.volread(tif_path, channels=channel)
        if color == 'Segmentation':
            ims_file.add_channel(data=volume, color_mode="TableColor", channel_name=name)
        else:
            ims_file.add_channel(data=volume, color_value=color, channel_name=name)


def main():
    with readers.get_volume_reader("data/instance/evaluation/T6/images/"
                                   "#6T S1 1024 crop_E4 0.5um_dapi.tif") \
            as reader, ImsFile("clb/dataprep/imaris/test.ims",
                               mode='x') as ims_output:
        img = np.squeeze(reader[..., 2].to_numpy())
        ims_output.add_channel(img, color_mode='BaseColor', color_value="Blue")


if __name__ == '__main__':
    main()
