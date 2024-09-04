import os
import pytest
import shutil
import tempfile
import unittest

import numpy as np

import clb.dataprep.imaris.ims_file as imsfile
import clb.dataprep.readers as readers
import clb.dataprep.utils as utils


@pytest.mark.io
class TestImsManip(unittest.TestCase):
    def setUp(self):
        save_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, save_dir)
        self.tif_path = os.path.join(save_dir, 'tiffile.tif')
        self.ims_path = os.path.join(save_dir, 'imsfile.ims')
        self.extracted_path = os.path.join(save_dir, 'extracted.tif')

    def test_read_and_write(self):
        ones = np.ones((5, 5))
        data = np.array([ones, 2 * ones, 3 * ones])[..., np.newaxis]
        utils.save_to_tif(self.tif_path, data)

        imsfile.add_channel_from_tif(self.ims_path, self.tif_path, 'Blue',
                                     'Test')
        imsfile.extract_channels_to_tif(self.ims_path, -1, self.extracted_path)
        with readers.get_volume_reader(self.extracted_path) as reader:
            np.testing.assert_equal(reader, data)
