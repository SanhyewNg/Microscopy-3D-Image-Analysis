import os
import shutil
import pytest
import unittest
import random
import string

import h5py
import numpy as np
import numpy.testing as nptest

from clb.dataprep.imaris.ims_file import ImsFile
from clb.dataprep.imaris.utils import load_attrs, update_attrs

@pytest.mark.io
class TestImarisWrapper(unittest.TestCase):
    def setUp(self):

        # create temp folder
        os.makedirs('./tmp', exist_ok=True)
        self.filepath = os.path.join('./tmp', 'test.ims')
        self.wrapper = ImsFile(self.filepath, "x")
        self.ims_file = h5py.File(self.filepath, "r")
        self.image_0 = np.random.randint(0, 256, (8, 200, 200))
        self.image_1 = np.random.randint(0, 256, (8, 200, 200))

    def test_FileIms_creates_file_in_path(self):

        self.assertTrue(os.path.isfile(self.filepath))

    def test_adding_channels(self):
        self.wrapper.add_channel(self.image_0, color_mode='BaseColor',
                                 color_value='Red')
        path_image0 = '/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'
        nptest.assert_array_equal(self.image_0,
                                  self.ims_file[path_image0][:, :, :])

    def test_channels_not_overlapping(self):
        self.wrapper.add_channel(self.image_0, color_mode='BaseColor',
                                 color_value='Red')
        path_image0 = '/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'

        self.wrapper.add_channel(self.image_1, color_mode='BaseColor',
                                 color_value='Red')
        path_image1 = '/DataSet/ResolutionLevel 0/TimePoint 0/Channel 1/Data'

        self.assertFalse(np.array_equal(self.ims_file[path_image0][:, :, :],
                                        self.ims_file[path_image1][:, :, :]))

    def test_load_attrs(self):
        input_dict = {"Attr_int": random.randint(0, 65535),
                      "Attr_float": random.random(),
                      "Attr_str": ''.join(
                          random.choice(string.ascii_uppercase + string.digits)
                          for _ in range(10))
                      }
        test_dict = {}
        for key, value in input_dict.items():
            test_dict[key] = str(value)
        test_grp = self.ims_file.create_group("Test_Group")
        update_attrs(test_grp, input_dict)
        output_dict = load_attrs(test_grp)

        for key in test_dict.keys():
            nptest.assert_array_equal(test_dict[key], output_dict[key])

    def tearDown(self):
        self.wrapper.close()
        self.ims_file.close()
        shutil.rmtree('./tmp')
