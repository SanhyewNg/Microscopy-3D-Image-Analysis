import os
import pytest
import shutil
import unittest
import unittest.mock as mock
from functools import partial

import imageio
import numpy as np
import numpy.testing as nptest

from clb.dataprep.utils import (add_padding, calc_desired_shape, ensure_4d,
                                fix_data_dim, fix_data_dims, load_tiff_stack,
                                remove_padding, find_touching_boundaries,
                                pick_channel, ensure_3d_rgb, ensure_2d_rgb)


class TestDataprepUtils(unittest.TestCase):
    def setUp(self):
        # create temp folder
        os.makedirs('./tmp', exist_ok=True)

        # create random input data with a square
        self.multi_bw_img = (np.random.random((200, 200)) * 128). \
            astype(np.uint8)
        self.multi_bw_img[80:120, 80:120] = 255
        imageio.mimsave(os.path.join('./tmp', 'tmp_bw.tif'),
                        [self.multi_bw_img, self.multi_bw_img])

        self.multi_multi_img = (np.random.random(
            (200, 200, 3)) * 128).astype(np.uint8)
        self.multi_multi_img[80:120, 80:120, :] = 255
        imageio.mimsave(os.path.join('./tmp', 'tmp_multi.tif'),
                        [self.multi_multi_img, self.multi_multi_img])

    @mock.patch('imageio.volread')
    @mock.patch('clb.dataprep.utils.get_number_of_pages')
    def test_single_stack_returned_single_page_grayscale_image(self,
                                                               pages_mock,
                                                               volread_mock):
        image = 100 * np.ones((100, 100), dtype=np.uint8)

        volread_mock.return_value = image
        pages_mock.return_value = 1

        loaded_img = load_tiff_stack('')
        self.assertEqual(loaded_img.shape, (1, 100, 100))

    @mock.patch('imageio.volread')
    @mock.patch('clb.dataprep.utils.get_number_of_pages')
    def test_correct_stack_returned_single_page_multichannel_image(self,
                                                                   pages_mock,
                                                                   volread_mock):
        image = 100 * np.ones((100, 100, 3), dtype=np.uint8)

        volread_mock.return_value = image
        pages_mock.return_value = 1

        loaded_img = load_tiff_stack('')
        self.assertEqual(loaded_img.shape, (1, 100, 100))

    def test_correct_stack_returned_multi_page_grayscale_image(self):
        loaded_img = load_tiff_stack('./tmp/tmp_bw.tif')
        self.assertEqual(loaded_img.shape, (2, 200, 200))

    def test_correct_stack_returned_multi_page_multi_channel_image(self):
        loaded_img = load_tiff_stack('./tmp/tmp_multi.tif')
        self.assertEqual(loaded_img.shape, (2, 200, 200))

    def test_correct_channel_returned_multi_page_mutli_channel_image(self):
        loaded_img_0 = load_tiff_stack('./tmp/tmp_multi.tif',
                                       partial(pick_channel, 0))
        self.assertEqual(loaded_img_0.shape, (2, 200, 200))

        nptest.assert_array_equal(self.multi_multi_img[::, ::, 0],
                                  loaded_img_0[0])
        loaded_img_1 = load_tiff_stack('./tmp/tmp_multi.tif',
                                       partial(pick_channel, 1))
        self.assertEqual(loaded_img_1.shape, (2, 200, 200))
        nptest.assert_array_equal(self.multi_multi_img[::, ::, 1],
                                  loaded_img_1[0])

    def test_fix_data_dims(self):
        input_size = 200
        desired_size = 256
        image = np.random.random((input_size, input_size))

        fixed_image = fix_data_dim(image, 'padding', desired_size)
        backed_image = fix_data_dim(fixed_image, 'padding', input_size)
        nptest.assert_array_equal(image, backed_image)

        fixed_image = fix_data_dim(image, 'reflect', desired_size)
        backed_image = fix_data_dim(fixed_image, 'reflect', input_size)
        nptest.assert_array_equal(image, backed_image)

        fixed_image = fix_data_dim(image, 'resize', desired_size)
        backed_image = fix_data_dim(fixed_image, 'resize', input_size)
        self.assertEqual(image.shape, backed_image.shape)

    def test_find_touching_boundaries(self):
        gt1 = np.array([[2, 0, 0, 3],
                        [2, 2, 3, 3],
                        [2, 2, 3, 3],
                        [2, 0, 0, 3]])
        boundaries1 = np.array([[0, 0, 0, 0],
                                [0, 1, 1, 0],
                                [0, 1, 1, 0],
                                [0, 0, 0, 0]])
        gt2 = np.array([[2, 0, 0, 3],
                        [0, 4, 4, 0],
                        [0, 4, 4, 0],
                        [2, 0, 0, 3]])
        boundaries2 = np.array([[1, 0, 0, 1],
                                [0, 1, 1, 0],
                                [0, 1, 1, 0],
                                [1, 0, 0, 1]])
        gt3 = np.array([[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]])
        boundaries3 = np.zeros((4, 4))

        output1 = find_touching_boundaries(gt1)
        output2 = find_touching_boundaries(gt2)
        output3 = find_touching_boundaries(gt3)
        nptest.assert_equal(output1, boundaries1)
        nptest.assert_equal(output2, boundaries2)
        nptest.assert_equal(output3, boundaries3)

    def test_add_remove_padding(self):
        image = np.random.random((200, 200))
        padded = add_padding(image)
        unpadded = remove_padding(padded, 200)
        np.array_equal(image, unpadded)

    def test_fix_data_dims_not_change_type(self):
        img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        gt = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        mod_img, mod_gt = fix_data_dims([img], [gt], 'padding', 256)

        self.assertTrue(mod_img.dtype == img.dtype)
        self.assertTrue(mod_gt.dtype == gt.dtype)

    def test_pick_channel_grayscale_image(self):
        single_channel = pick_channel(0, self.multi_bw_img)
        self.assertEqual(single_channel.shape, (200, 200))
        nptest.assert_array_equal(self.multi_bw_img, single_channel)

    def test_pick_channel_multi_channel_image(self):
        single_channel = pick_channel(1, self.multi_multi_img)
        self.assertEqual(single_channel.shape, (200, 200))
        nptest.assert_array_equal(self.multi_multi_img[::, ::, 1],
                                  single_channel)

    def tearDown(self):
        shutil.rmtree('./tmp')


class TestCalcDesiredShape(unittest.TestCase):
    def test_if_output_is_right(self):
        shape = (10, 10)
        current_1 = 0.5
        desired_1 = 1.
        current_x = 0.2
        current_y = 0.8
        desired_x = 1.
        desired_y = 1.

        resize_shape_1 = calc_desired_shape(shape,
                                            current_x=current_1,
                                            current_y=current_1,
                                            desired_x=desired_1,
                                            desired_y=desired_1)
        expected_shape_1 = (5, 5)
        resize_shape_xy = calc_desired_shape(shape,
                                             current_x=current_x,
                                             current_y=current_y,
                                             desired_x=desired_x,
                                             desired_y=desired_y)
        expected_shape_xy = (8, 2)

        self.assertEqual(resize_shape_1, expected_shape_1)
        self.assertEqual(resize_shape_xy, expected_shape_xy)


class TestEnsure2D(unittest.TestCase):
    def setUp(self):
        np.random.seed(14)

    def test_ensure_2d_rgb_from_2d(self):
        volume = np.random.random((7, 3))

        reshaped_image = ensure_2d_rgb(volume)
        self.assertEqual((7, 3, 3), reshaped_image.shape)
        np.testing.assert_equal(volume, reshaped_image[..., 0])
        self.assertEqual(0, reshaped_image[..., 1].sum() + reshaped_image[..., 2].sum())

    def test_ensure_2d_rgb_from_2dc2(self):
        volume = np.random.random((7, 3, 2))

        reshaped_image = ensure_2d_rgb(volume)
        self.assertEqual((7, 3, 3), reshaped_image.shape)
        np.testing.assert_equal(volume[..., 0], reshaped_image[..., 0])
        np.testing.assert_equal(volume[..., 1], reshaped_image[..., 1])
        self.assertEqual(0, reshaped_image[..., 2].sum())


class TestEnsure3D(unittest.TestCase):
    def setUp(self):
        np.random.seed(14)

    def test_ensure_3d_rgb_from_3d(self):
        volume = np.random.random((5, 7, 3))

        reshaped_image = ensure_3d_rgb(volume)
        self.assertEqual((5, 7, 3, 3), reshaped_image.shape)
        np.testing.assert_equal(volume, reshaped_image[..., 0])
        self.assertEqual(0, reshaped_image[..., 1].sum() + reshaped_image[..., 2].sum())

    def test_ensure_3d_rgb_from_4dc1(self):
        volume = np.random.random((5, 7, 3, 1))

        reshaped_image = ensure_3d_rgb(volume)
        self.assertEqual((5, 7, 3, 3), reshaped_image.shape)
        np.testing.assert_equal(volume[..., 0], reshaped_image[..., 0])
        self.assertEqual(0, reshaped_image[..., 1].sum() + reshaped_image[..., 2].sum())

    def test_ensure_3d_rgb_from_4dc2(self):
        volume = np.random.random((5, 7, 3, 2))

        reshaped_image = ensure_3d_rgb(volume)
        self.assertEqual((5, 7, 3, 3), reshaped_image.shape)
        np.testing.assert_equal(volume[..., 0], reshaped_image[..., 0])
        np.testing.assert_equal(volume[..., 1], reshaped_image[..., 1])
        self.assertEqual(0, reshaped_image[..., 2].sum())


class TestEnsure4D(unittest.TestCase):
    def test_2d_image(self):
        image = np.zeros((2, 2))

        reshaped_image = ensure_4d(image)
        expected_image = np.zeros((1, 2, 2, 1))

        np.testing.assert_equal(reshaped_image, expected_image)

    def test_3d_image_with_one_z(self):
        image = np.zeros((1, 2, 2))

        reshaped_image = ensure_4d(image)
        expected_image = np.zeros((1, 2, 2, 1))

        np.testing.assert_equal(reshaped_image, expected_image)

    def test_3d_image_with_two_zs_missing_channel(self):
        image = np.zeros((2, 3, 3))

        reshaped_image = ensure_4d(image)
        expected_image = np.zeros((2, 3, 3, 1))

        np.testing.assert_equal(reshaped_image, expected_image)

    def test_3d_image_with_two_cs_missing_z(self):
        image = np.zeros((3, 3, 2))

        reshaped_iamge = ensure_4d(image, assume_missing_channel=False)
        expected_image = np.zeros((1, 3, 3, 2))

        np.testing.assert_equal(reshaped_iamge, expected_image)

    def test_4d_image(self):
        image = np.zeros((2, 3, 4, 5))

        reshaped_image = ensure_4d(image)
        expected_image = np.zeros((2, 3, 4, 5))

        np.testing.assert_equal(reshaped_image, expected_image)
