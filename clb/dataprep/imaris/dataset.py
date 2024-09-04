import numpy as np
from scipy.ndimage.interpolation import zoom

from clb.dataprep.imaris.utils import update_attrs, load_attrs


class DataSet:
    def __init__(self, parent_ims, mode, resolution=None):
        """Creates a new dataset with its subclasses.

        Args:
            parent_ims (obj): A file object from h5py module.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
            resolution (int, none): Number of resolution levels
                inside the dataset.
        """
        if mode == "x":
            self.levels = resolution
            self.ims_group = parent_ims.create_group("DataSet")
        elif mode == "r+":
            self.ims_group = parent_ims["DataSet"]
            self.levels = len(self.ims_group.keys())

        self.resolution_list = [ResolutionLevel(self.ims_group, mode, level)
                                for level in range(self.levels)]

    def add_channel(self, data):
        """Passes channel data to the resolution_level objects,
        filling in the data and attrs.

        Args:
            data (arr): numpy array of shape (height, width, depth) containing
                a 3D image
        """
        for resolution_level in self.resolution_list:
            resolution_level.add_channel(data)


class ResolutionLevel:
    def __init__(self, parent_ims, mode, level=0):
        """Creates a new resolution level with its subclasses.

        Args:
            parent_ims (obj): A parent object from h5py module.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read.
            level (int): The resolution id number.
        """
        self.resolution_level = level
        self.group_name = "ResolutionLevel " + str(level)
        
        if mode == "x":
            self.ims_group = parent_ims.create_group(self.group_name)
        elif mode == "r+":
            self.ims_group = parent_ims[self.group_name]
        
        self.timepoint = TimePoint(self.ims_group, mode)

    def add_channel(self, data):
        """Passes channel data to the time points, correctly resized according
        to the resolution level the data is stored in.

        Args:
            data (arr): numpy array of shape (height, width, depth) containing
                a 3D image
        """

        resize_factor = 0.5 ** self.resolution_level
        resized_data = zoom(data, resize_factor)
        self.timepoint.add_channel(resized_data)


class TimePoint:
    def __init__(self, parent_ims, mode):
        """Creates a new timepoint with its subclasses.

        parent_ims (obj): A parent object from h5py module.
        mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
        """
        self.group_name = "TimePoint 0"
        self.channel_list = []

        if mode == "x":
            self.ims_group = parent_ims.create_group(self.group_name)
        elif mode == "r+":
            self.ims_group = parent_ims[self.group_name]
            for channel in self.ims_group.keys():
                self.channel_list += [Channel(self.ims_group,
                                              len(self.channel_list))]
                self.channel_list[-1].load_channel()

    def add_channel(self, data):
        """ Creates object of channel class assigned to the current TimePoint
        h5py object.

        Args:
            data (arr): numpy array of shape (height, width, depth) containing
                a 3D image
        """
        self.channel_list += [Channel(self.ims_group, len(self.channel_list))
                              ]
        self.channel_list[-1].add_channel(data)


class Channel:

    def __init__(self, parent_ims, num_channel):
        """Creates object of channel class assigned to the current Channel
        h5py object.

        Args:
            parent_ims (obj): A parent object from h5py module
            num_channel (int): ID number of a channel
        """
        self.num_channel = num_channel
        self.channel_name = "Channel " + str(self.num_channel)
        self.ims_group = parent_ims.require_group(self.channel_name)
        self.attrs = {}

    def load_channel(self):
        """Loads existing channel attributes.
        """
        self.attrs = load_attrs(self.ims_group)
        
    def add_channel(self, data):
        """Creates a new channel in a timepoint, passing data and attributes
        to .ims file, while storing only attributes.

        data (arr): numpy array of shape (Z, Y, X) containing
            a 3D image
        """
        min_bins = np.amin(data)
        max_bins = np.amax(data)
        self.attrs = {'ImageSizeZ': data.shape[0],
                      'ImageSizeY': data.shape[1],
                      'ImageSizeX': data.shape[2],
                      "HistogramMax": str("%.3f" % round(float(max_bins), 3)),
                      "HistogramMin": str("%.3f" % round(float(min_bins), 3)),
                      }
        self.store_data(self.ims_group, data)

        update_attrs(self.ims_group, self.attrs)

    def store_data(self, channel_ims, data):
        """ Stores given data in a .ims file

        Args:
            channel_ims (obj): object of a Group class from h5py module.
            data (arr): numpy array of shape (height, width, depth) containing
            a 3D image
        """
        channel_ims.create_dataset("Data", data=data, dtype=np.uint16,
                                   chunks=True, compression='gzip',
                                   compression_opts=7)
        self.get_histogram(channel_ims, data)

    @staticmethod
    def get_histogram(channel_ims, data):
        """ Makes a histogram from given data and stores it as a dataset in a
        .ims file

        Args:
            channel_ims (obj): object of a Group class from h5py module.
            data (arr): numpy array of shape (height, width, depth) containing
            a 3D image
        """
        histogram = np.histogram(data, 256)
        histogram_values = np.uint64(histogram[0])
        channel_ims.create_dataset("Histogram", data=histogram_values)
