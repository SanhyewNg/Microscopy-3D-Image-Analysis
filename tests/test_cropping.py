import os
import pytest
import unittest

import numpy as np

from clb.cropping import CropInfo
from clb.cropping.localize import find_position_2d, find_position_3d, find_positions_3d
from clb.cropping.volumeroi import VolumeROI
from clb.utils import channels_count


@pytest.mark.preprocessing
class TestLocalize(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

    def test_find_position_exact(self):
        image = np.random.random((200, 200))
        crop_image = image[37:66, 1:28]
        diff, top_left = find_position_2d(image, crop_image)
        self.assertEqual(0, diff)
        self.assertEqual((37, 1), top_left)

    def test_find_position_with_noise(self):
        image = np.random.random((200, 200))
        crop_image = image[37:66, 1:28]
        crop_image_with_noise = crop_image + np.random.random(crop_image.shape) / 20.0
        diff, top_left = find_position_2d(image, crop_image_with_noise)
        self.assertNotEqual(0, diff)
        self.assertEqual((37, 1), top_left)

    def test_find_position_3d_exact(self):
        volume = np.random.random((6, 20, 20))
        crop_image = volume[3, 3:17, 4:8]
        diff, crop_info = find_position_3d(volume, crop_image)
        self.assertEqual(0, diff)
        self.assertEqual(crop_image.shape, crop_info.shape)
        self.assertEqual(3, crop_info.y)
        self.assertEqual(4, crop_info.x)
        self.assertEqual(3, crop_info.z)

    def test_find_position_3d_with_noise(self):
        volume = np.random.random((6, 20, 20))
        crop_image = volume[3, 3:17, 4:8]
        crop_image_with_noise = crop_image + np.random.random(crop_image.shape) / 20.0
        diff, crop_info = find_position_3d(volume, crop_image_with_noise)
        self.assertNotEqual(0, diff)
        self.assertEqual(crop_image.shape, crop_info.shape)
        self.assertEqual(3, crop_info.y)
        self.assertEqual(4, crop_info.x)
        self.assertEqual(3, crop_info.z)

    def test_find_positions_3d_exact(self):
        volume = np.random.random((10, 20, 20))
        crop_volume = np.zeros((3, 14, 4))
        crop_volume[0] = volume[3, 3:17, 4:8]
        crop_volume[1] = volume[5, 3:17, 4:8]
        crop_volume[2] = volume[7, 3:17, 4:8]
        diff, crop_infos = find_positions_3d(volume, crop_volume)
        self.assertEqual(0, diff)
        self.assertEqual(3, len(crop_infos))
        self.assertEqual(crop_volume[0].shape, crop_infos[0].shape)
        self.assertEqual(crop_infos[0].y, crop_infos[1].y)
        self.assertEqual(crop_infos[1].y, crop_infos[2].y)
        self.assertEqual(crop_infos[0].x, crop_infos[1].x)
        self.assertEqual(crop_infos[1].x, crop_infos[2].x)
        self.assertEqual(3, crop_infos[0].z)
        self.assertEqual(5, crop_infos[1].z)
        self.assertEqual(7, crop_infos[2].z)

    def test_find_positions_3d_with_noise(self):
        volume = np.random.random((10, 20, 20))
        crop_volume = np.zeros((3, 14, 4))
        crop_volume[0] = volume[3, 3:17, 4:8]
        crop_volume[1] = volume[5, 3:17, 4:8]
        crop_volume[2] = volume[7, 3:17, 4:8]
        crop_image_with_noise = crop_volume + np.random.random(crop_volume.shape) / 20.0
        diff, crop_infos = find_positions_3d(volume, crop_image_with_noise)
        self.assertNotEqual(0, diff)
        self.assertEqual(3, len(crop_infos))
        self.assertEqual(crop_volume[0].shape, crop_infos[0].shape)
        self.assertEqual(crop_infos[0].y, crop_infos[1].y)
        self.assertEqual(crop_infos[1].y, crop_infos[2].y)
        self.assertEqual(crop_infos[0].x, crop_infos[1].x)
        self.assertEqual(crop_infos[1].x, crop_infos[2].x)
        self.assertEqual(3, crop_infos[0].z)
        self.assertEqual(5, crop_infos[1].z)
        self.assertEqual(7, crop_infos[2].z)

    def test_find_positions_3d_non_arithmetic(self):
        volume = np.random.random((10, 20, 20))
        crop_volume = np.zeros((3, 14, 4))
        crop_volume[0] = volume[3, 3:17, 4:8]
        crop_volume[1] = volume[6, 3:17, 4:8]
        crop_volume[2] = volume[7, 3:17, 4:8]

        with self.assertRaises(AssertionError) as err:
            _ = find_positions_3d(volume, crop_volume)


@pytest.mark.preprocessing
class TestCropInfo(unittest.TestCase):
    tmp_file_path = "tmp.yaml"

    def setUp(self):
        np.random.seed(10)

    def test_crop_info_is_block(self):
        volume = CropInfo.create_volume(3, 3, 10, 10, [-1, 0, 1, 2])
        self.assertEqual(True, CropInfo.is_block(volume))

        crop_infos = []
        self.assertEqual(None, CropInfo.is_block(crop_infos))
        crop_infos.append(CropInfo(10, 10, 5, (3, 4)))
        self.assertEqual(True, CropInfo.is_block(crop_infos))
        crop_infos.append(CropInfo(10, 10, 6, (3, 4)))
        self.assertEqual(True, CropInfo.is_block(crop_infos))

        # duplicate
        self.assertEqual(False, CropInfo.is_block(crop_infos + [CropInfo(10, 10, 6, (3, 4))]))
        # bad order
        self.assertEqual(False, CropInfo.is_block(crop_infos + [CropInfo(10, 10, 4, (3, 4))]))
        # gap
        self.assertEqual(False, CropInfo.is_block(crop_infos + [CropInfo(10, 10, 8, (3, 4))]))
        # bad x
        self.assertEqual(False, CropInfo.is_block(crop_infos + [CropInfo(10, 11, 7, (3, 4))]))
        # bad shape
        self.assertEqual(False, CropInfo.is_block(crop_infos + [CropInfo(10, 11, 7, (4, 3))]))

        self.assertEqual(True, CropInfo.is_block(crop_infos + [CropInfo(10, 10, 7, (3, 4))]))

    def test_crop_info_block_size(self):
        volume = CropInfo.create_volume(3, 3, 10, 10, [-1, 0, 1, 2])
        self.assertEqual((4, 10, 10), CropInfo.block_size(volume))

        crop_infos = []
        self.assertEqual(None, CropInfo.block_size(crop_infos))
        crop_infos.append(CropInfo(10, 10, 5, (3, 4)))
        self.assertEqual((1, 3, 4), CropInfo.block_size(crop_infos))
        crop_infos.append(CropInfo(10, 10, 6, (3, 4)))
        self.assertEqual((2, 3, 4), CropInfo.block_size(crop_infos))

        self.assertEqual((3, 3, 4), CropInfo.block_size(crop_infos + [CropInfo(10, 10, 7, (3, 4))]))

        with self.assertRaises(ValueError):
            CropInfo.block_size(crop_infos + [CropInfo(10, 11, 7, (4, 3))])

    def test_crop_info_save_load(self):
        crop_info1 = CropInfo(10, 20, 3, (5, 6))
        crop_info2 = CropInfo(10, 33, 4, (6, 7))
        CropInfo.save([crop_info1, crop_info2], self.tmp_file_path)

        loaded = CropInfo.load(self.tmp_file_path)
        self.assertEqual(str(crop_info1), str(loaded[0]))
        self.assertEqual(False, crop_info1 is loaded[0])
        self.assertEqual(str(crop_info2), str(loaded[1]))
        self.assertEqual(False, crop_info2 is loaded[1])

    def test_crop_info_restrict_outside_zs(self):
        crop_info1 = CropInfo(10, 20, 3, (5, 6))
        crop_info2 = CropInfo(10, 20, -3, (5, 6))

        self.assertEqual(crop_info1, crop_info1.restrict((4, 100, 100)))

        self.assertEqual(None, crop_info2.restrict((4, 100, 100)))
        self.assertEqual(None, crop_info1.restrict((3, 100, 100)))

    def test_crop_info_restrict_outside_xs(self):
        crop_info1 = CropInfo(10, -1, 10, (10, 5))
        crop_info2 = CropInfo(10, 20, 10, (10, 15))

        self.assertEqual(CropInfo(10, 0, 10, (10, 4)), crop_info1.restrict((100, 100, 40)))
        self.assertEqual(crop_info2, crop_info2.restrict((100, 100, 40)))

        self.assertEqual(CropInfo(10, 20, 10, (10, 5)), crop_info2.restrict((100, 100, 25)))
        self.assertEqual(None, crop_info2.restrict((100, 100, 20)))

    def test_crop_info_restrict_outside_ys(self):
        crop_info1 = CropInfo(-1, 10, 10, (5, 10))
        crop_info2 = CropInfo(20, 10, 10, (15, 10))

        self.assertEqual(CropInfo(0, 10, 10, (4, 10)), crop_info1.restrict((100, 40, 100)))
        self.assertEqual(crop_info2, crop_info2.restrict((100, 40, 100)))

        self.assertEqual(CropInfo(20, 10, 10, (5, 10)), crop_info2.restrict((100, 25, 100)))
        self.assertEqual(None, crop_info2.restrict((100, 20, 100)))

    def test_crop_infos_restrict(self):
        crop_info1 = CropInfo(10, 20, 3, (5, 6))
        crop_info2 = CropInfo(10, 40, 3, (5, 6))  # drop it
        crop_info3 = CropInfo(15, 20, 4, (7, 6))
        crop_info4 = CropInfo(10, -6, 4, (5, 9))
        crop_info5 = CropInfo(-5, -6, 4, (5, 9))  # drop it
        crop_info6 = CropInfo(5, 6, -1, (5, 9))  # drop it

        crop_infos = [crop_info1, crop_info2, crop_info3, crop_info4, crop_info5, crop_info6]
        restricted = CropInfo.restrict_infos(crop_infos, (5, 20, 30))

        self.assertEqual(CropInfo(10, 20, 3, (5, 6)), restricted[0])
        self.assertEqual(CropInfo(15, 20, 4, (5, 6)), restricted[1])
        self.assertEqual(CropInfo(10, 0, 4, (5, 3)), restricted[2])

        self.assertEqual(3, len(restricted))

    def test_properties(self):
        crop_info1 = CropInfo(10, 20, 3, (5, 6))
        self.assertEqual(30, crop_info1.area)
        self.assertEqual(26, crop_info1.x_end)
        self.assertEqual(15, crop_info1.y_end)

    def test_overlap_iou(self):
        crop_info1 = CropInfo(10, 20, 3, (5, 6))
        crop_info2 = CropInfo(10, 20, 4, (5, 6))
        self.assertEqual(0, crop_info1.overlap(crop_info2))
        self.assertEqual(0, CropInfo.iou_volume([crop_info1], [crop_info2]))
        self.assertEqual(0, CropInfo.overlap_volume_fraction([crop_info1], [crop_info2]))

        crop_info2.z = 3
        self.assertEqual(30, crop_info1.overlap(crop_info2))
        self.assertEqual(30, CropInfo.overlap_volume([crop_info1], [crop_info2]))
        self.assertEqual(1.0, CropInfo.iou_volume([crop_info1], [crop_info2]))
        self.assertEqual(1.0, CropInfo.overlap_volume_fraction([crop_info1], [crop_info2]))

        crop_info2.x = 21
        crop_info2.y = 9
        self.assertEqual(20, crop_info1.overlap(crop_info2))
        self.assertEqual(20, CropInfo.overlap_volume([crop_info1], [crop_info2]))
        self.assertEqual(0.5, CropInfo.iou_volume([crop_info1], [crop_info2]))
        self.assertEqual(20 / 30, CropInfo.overlap_volume_fraction([crop_info1], [crop_info2]))

        crop_info3 = CropInfo(14, 18, 3, (5, 5))
        # 1 x 3
        self.assertEqual(3, crop_info1.overlap(crop_info3))
        self.assertEqual(3, CropInfo.overlap_volume([crop_info1], [crop_info3]))
        self.assertEqual(3 / (55 - 3), CropInfo.iou_volume([crop_info1], [crop_info3]))
        self.assertEqual(3 / 30, CropInfo.overlap_volume_fraction([crop_info1], [crop_info3]))
        self.assertEqual(3 / 25, CropInfo.overlap_volume_fraction([crop_info3], [crop_info1]))

        crop_info4 = CropInfo(16, 18, 3, (5, 5))
        # 0 x 3
        self.assertEqual(0, crop_info1.overlap(crop_info4))
        self.assertEqual(0, CropInfo.iou_volume([crop_info1], [crop_info4]))

    def test_centered_crops_creation(self):
        with self.assertRaises(AssertionError):
            CropInfo.create_centered_volume((3, 3), (3, 3))

        crop_infos = CropInfo.create_centered_volume((5, 6, 7), (3, 10, 15))
        self.assertEqual(3, len(crop_infos))
        self.assertEqual(CropInfo(1, 0, 4, (10, 15)), crop_infos[0])
        self.assertEqual(CropInfo(1, 0, 5, (10, 15)), crop_infos[1])
        self.assertEqual(CropInfo(1, 0, 6, (10, 15)), crop_infos[2])

    def test_crops_near_the_edges(self):
        crop_shape = (9, 10, 8)
        crop_infos = CropInfo.create_centered_volume((3, 2, 7), crop_shape)
        self.assertEqual(9, len(crop_infos))
        self.assertEqual(CropInfo(-3, 3, -1, crop_shape[1:]), crop_infos[0])
        self.assertEqual(CropInfo(-3, 3, 7, crop_shape[1:]), crop_infos[-1])

    def test_extends_info(self):
        crop_infos = CropInfo.create_volume(y=3, x=3, height=2, width=2, zs=[0,1,2])
        extended = CropInfo.extend_infos(crop_infos, (2, 1, 4))
        self.assertEqual(3+4, len(extended))
        self.assertEqual(True, CropInfo.is_block(extended))
        self.assertEqual(CropInfo(z=-2, y=2, x=-1, shape=(4, 10)), extended[0])
        self.assertEqual(CropInfo(z=4, y=2, x=-1, shape=(4, 10)), extended[-1])

        reduced_to_same = CropInfo.extend_infos(extended, (-2, -1, -4))
        self.assertEqual(crop_infos, reduced_to_same)

    def test_extends_negative(self):
        crop_infos = CropInfo.create_volume(y=3, x=3, height=5, width=4, zs=[0, 1, 2, 3])
        reduced_to_small = CropInfo.extend_infos(crop_infos, (-1, -2, -1))

        self.assertEqual(2, len(reduced_to_small))
        self.assertEqual(True, CropInfo.is_block(reduced_to_small))
        self.assertEqual(CropInfo(z=1, y=5, x=4, shape=(1, 2)), reduced_to_small[0])
        self.assertEqual(CropInfo(z=2, y=5, x=4, shape=(1, 2)), reduced_to_small[-1])

    def tearDown(self):
        if os.path.isfile(self.tmp_file_path):
            os.remove(self.tmp_file_path)


@pytest.mark.preprocessing
class TestVolumeROI(unittest.TestCase):
    tmp_file_path = "tmp.yaml"

    def setUp(self):
        np.random.seed(10)

    def test_create_empty(self):
        crop_infos = CropInfo.create_centered_volume((3, 2, 7), (3, 6, 3))
        roi = VolumeROI.create_empty(crop_infos, np.uint16)
        self.assertEqual((3, 6, 3), roi.shape)
        self.assertEqual(np.uint16, roi.crop_volume.dtype)

    def test_invalid_crop_info(self):
        data = np.random.random((3, 3, 3))
        crop_infos = CropInfo.create_centered_volume((3, 2, 7), (3, 3, 3))
        proper_roi = VolumeROI(crop_infos, data)

        with self.assertRaises(ValueError):
            invalid_crop_infos = crop_infos + [CropInfo(1, 1, 5, (3, 3, 3))]
            invalid_roi = VolumeROI(invalid_crop_infos, data)

    def test_slicing_absolute(self):
        data = np.random.random((9, 6, 8))
        crop_infos = CropInfo.create_volume(3, -5, 6, 8, [-2, -1, 0, 1, 2, 3, 4, 5, 6])
        proper_roi = VolumeROI(crop_infos, data)

        self.assertEqual((-2, 7), proper_roi.span_intesect(proper_roi, 0))
        self.assertEqual(slice(0, 1), proper_roi.slice_from_absolute(0, (-5, -1)))
        self.assertEqual(slice(0, 5), proper_roi.slice_from_absolute(0, (-4, 3)))
        self.assertEqual(slice(1, 5), proper_roi.slice_from_absolute(0, (-1, 3)))
        self.assertEqual(slice(1, 8), proper_roi.slice_from_absolute(0, (-1, 6)))
        self.assertEqual(slice(1, 12), proper_roi.slice_from_absolute(0, (-1, 10)))
        self.assertEqual(slice(10, 12), proper_roi.slice_from_absolute(0, (8, 10)))

        self.assertEqual(slice(0, 2), proper_roi.slice_from_absolute(1, (2, 5)))
        self.assertEqual(slice(5, 15), proper_roi.slice_from_absolute(2, (0, 10)))

        absolute_spans = [(2, 3), (3, 4), (-4, -3)]
        single_item = proper_roi.extract_absolute(absolute_spans)
        self.assertEqual(1, single_item.size)
        self.assertEqual(data[4, 0, 1], single_item[0, 0, 0])

    def test_crops_near_the_edges(self):
        data = np.random.random((10, 10, 10))

        crop_shape = (9, 10, 8)
        crop_infos = CropInfo.create_centered_volume((3, 2, 7), crop_shape)
        self.assertEqual(9, len(crop_infos))
        self.assertEqual(CropInfo(-3, 3, -1, crop_shape[1:]), crop_infos[0])
        self.assertEqual(CropInfo(-3, 3, 7, crop_shape[1:]), crop_infos[-1])

        bounded_crop_infos = CropInfo.restrict_infos(crop_infos, data.shape)
        bounded_volume_image = CropInfo.crop_volume(data, bounded_crop_infos)
        bounded_roi = VolumeROI(bounded_crop_infos, bounded_volume_image)
        self.assertEqual((8, 7, 7), bounded_roi.shape)

        cell_roi = VolumeROI.create_empty(crop_infos, dtype=data.dtype, channels=channels_count(data))
        self.assertEqual(crop_shape, cell_roi.shape[:3])

        cell_roi.implant(bounded_roi)
        self.assertEqual(cell_roi.crop_volume.sum(), bounded_roi.crop_volume.sum())

        padded_roi = VolumeROI.from_absolute_crop_with_padding(crop_infos, data)
        self.assertEqual(cell_roi.crop_volume.sum(), padded_roi.crop_volume.sum())
        self.assertEqual(crop_shape, padded_roi.shape[:3])
        self.assertEqual(crop_shape, padded_roi.crop_volume.shape[:3])

    def test_crops_micro_cell_near_the_edges(self):
        crop_infos = CropInfo.create_centered_volume((0, 0, 0), (1, 1, 1))
        self.assertEqual(1, len(crop_infos))

        crop_infos = CropInfo.create_centered_volume((9, 9, 9), (1, 1, 1))
        self.assertEqual(1, len(crop_infos))

        crop_infos = CropInfo.create_centered_volume((0, 0, 0), (2, 2, 2))
        self.assertEqual(2, len(crop_infos))

        crop_infos = CropInfo.create_centered_volume((9, 9, 9), (2, 2, 2))
        self.assertEqual(2, len(crop_infos))

    def tearDown(self):
        if os.path.isfile(self.tmp_file_path):
            os.remove(self.tmp_file_path)
