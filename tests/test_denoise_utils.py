import unittest

from clb.denoising.utils import select_stride


class TestSelectStride(unittest.TestCase):
    def test_pass(self):
        self.assertTupleEqual(select_stride((512, 512)), (64,64)) 
        self.assertTupleEqual(select_stride((600, 600)), (43,43))

    def test_fail_when_image_size_grater_than_stride_size(self):
        with self.assertRaises(AssertionError):
            select_stride((200, 200), patch_size=(201, 201))

    def test_fail_when_max_stride_size_grater_than_patch_size(self):
        with self.assertRaises(AssertionError):
            select_stride((200, 200), (65,65), (64, 64))
