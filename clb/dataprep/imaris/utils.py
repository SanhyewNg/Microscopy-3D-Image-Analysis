import os

import h5py
import numpy as np


class GetImsPaths:
    def __init__(self, file):
        self.path_list = []
        self.file = file
        self.file.visit(self._get_recursive_paths)

    def _get_recursive_paths(self, path):
        """Works with file.visit() method from h5py module and feeds a
        recursive list of paths to the path_list field.
        Args:
            Currently added path.
        """
        path = ''.join(path)
        self.path_list += [path]

    def get_paths_keywords(self, keywords):
        """Get paths from path_list with specific keywords.

        Args:
            keywords (str/list): Keywords that must be included in the path.
        Returns:
            List of paths containing such keywords.
        """
        res = []
        for path in self.path_list:
            pathnorm = os.path.normpath(path)
            split_path = pathnorm.split(os.sep)

            if set(keywords).issubset(split_path):
                res += [path]
        return res

    def get_data_from_ims(self, channel_list=0):
        """Gets data from channels with ID number specified.

        Args:
            channel_list (int/list): an ID number or a list of ID numbers of the
                channels user wants data from.

        Returns:
            data_dict (dict): A dictionary with data from the channels.
        """

        keywords = ["ResolutionLevel 0", "Data"]

        if isinstance(channel_list, int):
            channel_list = [channel_list]

        keywords.extend('Channel ' + str(channel) for channel in channel_list)

        data_paths = self.get_paths_keywords(keywords)

        data_dict = {}
        for path in data_paths:
            pathnorm = os.path.normpath(path)
            split_path = pathnorm.split(os.sep)
            for channel_name in split_path:
                if "Channel" in channel_name:
                    data = self.file[path]
                    data_dict[channel_name] = data[:, :, :]


        return data_dict


def load_attrs(group_ims):
    """This method allows to load attributes from a given group into a
    dictionary

    Args:
        group_ims (obj): H5py Group obj
    Returns:
        attrs_dict (dict) A dictionary of attributes of that group
    """

    attrs_dict = {key: ''.join(value.astype(str)) for key, value in
                  group_ims.attrs.items()}

    return attrs_dict


def update_attrs(group_ims, attrs):
    """ This method allows given group to update its attributes
    Args:
        group_ims (obj): H5py Group obj
        attrs (dict): Dictionary of attributes of the given group.
    """
    for attribute, value in attrs.items():
        if isinstance(value, float) or isinstance(value, int):
            value = str(value)
        value = list(value)

        value = np.asarray(value, dtype="|S1")
        group_ims.attrs.create(attribute, value)


def main():
    f = h5py.File("clb/dataprep/imaris/test.ims", 'r')
    getd = GetImsPaths(f)
    path_list = getd.get_paths_keywords(['ResolutionLevel 0', 'Data'])
    print(path_list)
    data_dict = getd.get_data_from_ims(0)
    print(data_dict.keys())
    print(data_dict.values())


if __name__ == '__main__':
    main()
