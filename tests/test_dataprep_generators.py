import pytest
import unittest
import unittest.mock as mock
import warnings
from itertools import tee

import numpy as np
import numpy.testing as nptest

import clb.dataprep.augmentations2D as augmentations
from clb.dataprep.generators import (raw_data_generator, blobs_removal,
                                     add_bin_channel,
                                     add_boundaries_channel,
                                     batch_generator, form_dcan_input,
                                     form_tensors, normalizer,
                                     rescaler,
                                     single_image_generator,
                                     subtract_boundaries_from_objects,
                                     _map_readers_to_pages, _shuffle_pages)
from vendor.genny.genny.wrappers import obj_gen_wrapper


@pytest.mark.io
class TestGenerators(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        pass

    @obj_gen_wrapper
    def raw_gen_test(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 1

        values = [(img, gt), (img, gt), (img, gt), (img, gt)]

        for tup in values:
            yield tup

    def test__map_readers_to_pages(self):
        mock_im_reader = mock.MagicMock()
        mock_im_reader.__len__ = mock.Mock(return_value=3)
        mock_gt_reader = mock.MagicMock()
        mock_gt_reader.__len__ = mock.Mock(return_value=3)
        mock_get_reader = mock.Mock(side_effect=[mock_im_reader,
                                                 mock_gt_reader])
        with mock.patch('clb.dataprep.generators.get_reader',
                        new=mock_get_reader):
            mapping = _map_readers_to_pages(['im_data'], ['gt_data'])
            right_mapping = {(mock_im_reader, mock_gt_reader): [0, 1, 2]}

            self.assertEqual(mapping, right_mapping)

    def test__shuffle_pages_if_shuffles(self):
        lists = (
            [1, 2, 3],
            [4, 5, 6]
        )

        _shuffle_pages(lists, seed=1)

        shuffled_lists = (
            [2, 3, 1],
            [6, 4, 5]
        )

        self.assertEqual(lists, shuffled_lists)

    def test_raw_data_generator(self):
        zeros = np.zeros((2, 2))
        ones = np.ones((2, 2))
        twos = 2 * np.ones((2, 2))
        threes = 3 * np.ones((2, 2))
        fours = 4 * np.ones((2, 2))
        fives = 5 * np.ones((2, 2))
        img_in = np.array([zeros, ones, twos])
        gt_in = np.array([threes, fours, fives])
        mock_paths_reader = mock.Mock(side_effect=[['img'], ['gt']])
        mock_reader = mock.Mock(side_effect=[img_in, gt_in])

        with mock.patch('clb.dataprep.generators.'
                        'get_tiff_paths_from_directories',
                        new=mock_paths_reader), \
            mock.patch('clb.dataprep.generators.load_tiff_stack',
                       new=mock_reader):
            data_gen = raw_data_generator('path', 'path', infinite=False)
            right_output = (
                (zeros, threes),
                (ones, fours),
                (twos, fives)
            )

            nptest.assert_equal(tuple(data_gen), right_output)

    def test_raw_data_generator_multichannel(self):
        zeros = np.zeros((2, 2))
        ones = np.ones((2, 2))
        twos = 2 * np.ones((2, 2))
        threes = 3 * np.ones((2, 2))
        fours = 4 * np.ones((2, 2))
        fives = 5 * np.ones((2, 2))
        img_in = np.array([zeros, ones, twos, threes, fours, fives])
        gt_in = np.array([zeros, ones, twos, threes, fours, fives])
        mock_paths_reader = mock.Mock(side_effect=[['img'], ['gt']])
        mock_reader = mock.Mock(side_effect=[img_in, gt_in])

        with mock.patch('clb.dataprep.generators.'
                        'get_tiff_paths_from_directories',
                        new=mock_paths_reader), \
            mock.patch('clb.dataprep.generators.load_tiff_stack',
                       new=mock_reader):
            data_gen = raw_data_generator('path', 'path', 3, infinite=False,
                                          spatial_context=True)
            right_output = (
                            (np.stack((zeros, ones, twos), axis=2), ones),
                            (np.stack((ones, twos, threes), axis=2), twos),
                            (np.stack((twos, threes, fours), axis=2), threes),
                            (np.stack((threes, fours, fives), axis=2), fours),
            )

            nptest.assert_equal(tuple(data_gen), right_output)

    def test_remove_blobs(self):
        img = 3 * np.ones((3, 3))
        gt = np.array([[5, 1, 3], [1, 3, 2], [0, 9, 1]])
        blob_free_img_ones = np.array([[3, 0, 3], [0, 3, 3], [3, 3, 0]])
        blob_free_gt_ones = np.array([[5, 0, 3], [0, 3, 2], [0, 9, 0]])
        blob_free_img_threes = np.array([[3, 3, 0], [3, 0, 3], [3, 3, 3]])
        blob_free_gt_threes = np.array([[5, 1, 0], [1, 0, 2], [0, 9, 1]])

        output_img1, output_gt1 = next(blobs_removal([(img, gt)],
                                                     remove_blobs=True,
                                                     blob_marker=1))
        output_img3, output_gt3 = next(blobs_removal([(img, gt)],
                                                     remove_blobs=True,
                                                     blob_marker=3))
        output_img_false, output_gt_false = next(blobs_removal([(img, gt)],
                                                 remove_blobs=False,
                                                 blob_marker=3))

        nptest.assert_equal(output_img1, blob_free_img_ones)
        nptest.assert_equal(output_gt1, blob_free_gt_ones)
        nptest.assert_equal(output_img3, blob_free_img_threes)
        nptest.assert_equal(output_gt3, blob_free_gt_threes)
        nptest.assert_equal(output_img_false, img)
        nptest.assert_equal(output_gt_false, gt)

    def test_new_channel_added_when_add_bin_channel(self):
        gt = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        _, output = next(add_bin_channel([(None, gt)]))
        self.assertEqual(output.shape, (256, 256, 2))

    def test_type_kept_when_add_bin_channel(self):
        gt = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        _, output = next(add_bin_channel([(None, gt)]))
        self.assertEqual(output.dtype, gt.dtype)

    def test_gt_binarized_when_add_bin_channel(self):
        gt = np.zeros((16, 16), dtype=np.uint8)
        gt[4:8, 4:8] = 15

        _, output = next(add_bin_channel([(None, gt)]))

        nptest.assert_array_equal(output[0:4, 0:4, 1], np.zeros((4, 4)))
        nptest.assert_array_equal(output[4:8, 4:8, 1], 255 * np.ones((4, 4)))

    def test_new_channel_added_when_1_channel_and_add_boundaries_channel(self):
        gt = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        _, output = next(add_boundaries_channel([(None, gt)]))
        self.assertEqual(output.shape, (256, 256, 2))

    def test_type_kept_when_1_channel_and_add_boundaries_channel(self):
        gt = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        _, output = next(add_boundaries_channel([(None, gt)]))
        self.assertEqual(output.dtype, gt.dtype)

    def test_new_channel_added_when_3_channels_and_add_boundaries_channel(self):
        gt = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        _, output = next(add_boundaries_channel([(None, gt)]))
        self.assertEqual(output.shape, (256, 256, 4))

    def test_boundaries_subtracted_when_subtract_boundaries_from_objects(self):
        gt = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        gt_obj = add_bin_channel([(None, gt)])
        gt_obj_bnd = add_boundaries_channel(gt_obj)
        mod_gt_obj_bnd = \
            subtract_boundaries_from_objects(gt_obj_bnd)
        _, gt = next(mod_gt_obj_bnd)
        intersection = gt[..., 1] & gt[..., 2]

        nptest.assert_array_equal(intersection, np.zeros((256, 256)))

    def test_max_value_kept_when_subtract_boundaries_from_objects(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 255
        data_gen = [(img, gt), (img, gt), (img, gt), (img, gt)]

        gt_obj = add_bin_channel(data_gen)
        gt_obj_bnd = add_boundaries_channel(gt_obj)

        for _, gt in subtract_boundaries_from_objects(gt_obj_bnd):
            self.assertTrue(np.max(gt) == 255)

    def test_types_correct_when_normalizer_applied_uint8(self):
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        gt = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        norm_img, norm_gt = next(normalizer([(img, gt)]))

        self.assertEqual(norm_img.dtype, np.float64)
        self.assertEqual(norm_gt.dtype, np.uint8)

    def test_types_correct_when_normalizer_applied_uint16(self):
        img = np.random.randint(0, 65536, (256, 256), dtype=np.uint16)
        gt = np.random.randint(0, 65536, (256, 256), dtype=np.uint16)

        norm_img, norm_gt = next(normalizer([(img, gt)]))

        self.assertEqual(norm_img.dtype, np.float64)
        self.assertEqual(norm_gt.dtype, np.uint16)

    def test_values_normalized_when_normalizer_applied_uint8(self):
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        gt = np.zeros((256, 256), dtype=np.uint8)
        gt[25:75, 25:75] = 255

        norm_img, norm_gt = next(normalizer([(img, gt)]))
        self.assertTrue(np.max(norm_img) <= 1.0)
        self.assertTrue(np.min(norm_img) >= 0.0)
        self.assertTrue(np.max(norm_gt) == 1)
        self.assertTrue(np.min(norm_gt) == 0)

    def test_values_normalized_when_normalizer_applied_uint16(self):
        img = np.random.randint(0, 65536, (256, 256), dtype=np.uint16)
        gt = np.zeros((256, 256), dtype=np.uint16)
        gt[25:75, 25:75] = 65535

        norm_img, norm_gt = next(normalizer([(img, gt)]))
        self.assertTrue(np.max(norm_img) <= 1.0)
        self.assertTrue(np.min(norm_img) >= 0.0)
        self.assertTrue(np.max(norm_gt) == 1)
        self.assertTrue(np.min(norm_gt) == 0)

    def test_array_rescaled_when_rescaler_applied(self):
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        gt = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        mod_img, mod_gt = next(rescaler(data_gen=[(img, gt)],
                                        trim_method='padding',
                                        out_dim=512))

        nptest.assert_array_equal(mod_img.shape, (512, 512))
        nptest.assert_array_equal(mod_gt.shape, (512, 512))

    def test_array_rescaled_when_rescaler_applied_multichannel_resize(self):
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        gt = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        mod_img, mod_gt = next(rescaler(data_gen=[(img, gt)],
                                        trim_method='resize',
                                        out_dim=512))

        nptest.assert_array_equal(mod_img.shape, (512, 512, 3))
        nptest.assert_array_equal(mod_gt.shape, (512, 512, 3))

    def test_array_rescaled_when_rescaler_applied_multichannel_padding(self):
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        gt = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        mod_img, mod_gt = next(rescaler(data_gen=[(img, gt)],
                                        trim_method='resize',
                                        out_dim=512))

        nptest.assert_array_equal(mod_img.shape, (512, 512, 3))
        nptest.assert_array_equal(mod_gt.shape, (512, 512, 3))

    def test_type_kept_when_rescaler_applied(self):
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        gt = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        mod_img, mod_gt = next(rescaler(data_gen=[(img, gt)],
                                        trim_method='padding',
                                        out_dim=512))

        self.assertTrue(mod_img.dtype == img.dtype)
        self.assertTrue(mod_gt.dtype == gt.dtype)

    def test_warning_raised_when_out_dim_different_than_input_image(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 1
        data_gen = [(img, gt), (img, gt), (img, gt), (img, gt)]

        trim_method = 'resize'
        out_dim = 256

        rescale_gen = rescaler(data_gen=data_gen,
                               trim_method=trim_method,
                               out_dim=out_dim)

        warning_raised = False
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                _, _ = next(rescale_gen)
            except UserWarning:
                warning_raised = True

        self.assertTrue(warning_raised)

    def test_values_still_binary_when_rescaler_applied(self):
        trim_method = 'padding'
        out_dim = 100
        augs = 1
        seed = 43
        mode = 'constant'
        pad = None

        augmentator = augmentations.AugGenerator(pad=pad,
                                                 mode=mode,
                                                 seed=seed)

        gen_chain = (self.raw_gen_test() |
                     add_bin_channel(obj_value=255) |
                     add_boundaries_channel(bnd_value=255) |
                     subtract_boundaries_from_objects() |
                     augmentator.flow(augs=augs, ensure_gt_binary=True) |
                     rescaler(trim_method=trim_method, out_dim=out_dim))

        for _, mod_gt in gen_chain:
            self.assertTrue(len(np.unique(mod_gt[..., 1])) == 2)
            self.assertTrue(np.max(mod_gt[..., 1]) == 255)
            self.assertTrue(np.min(mod_gt[..., 1]) == 0)

    def test_tensors_formed_when_form_tensors_applied(self):
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        gt = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        img_tensor, gt_tensor = next(form_tensors(
            data_gen=[(img, gt)]))

        nptest.assert_array_equal(img_tensor.shape, (1, 256, 256, 1))
        nptest.assert_array_equal(gt_tensor.shape, (1, 256, 256, 1))

    def test_batch_formed_when_batch_generator_applied(self):
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        gt = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        img_tensor, gt_tensor = next(form_tensors(
            data_gen=[(img, gt)]))

        img_batch, gt_batch = next(batch_generator(
            data_gen=[(img_tensor, gt_tensor),
                      (img_tensor, gt_tensor),
                      (img_tensor, gt_tensor)],
            batch_size=3))

        nptest.assert_array_equal(img_batch.shape, (3, 256, 256, 1))
        nptest.assert_array_equal(gt_batch.shape, (3, 256, 256, 1))

    def test_valid_input_formed_for_dcan_when_form_dcan_input_applied(self):
        trim_method = 'padding'
        out_dim = 256
        augs = 1
        seed = 43
        mode = 'constant'
        pad = None

        augmentator = augmentations.AugGenerator(pad=pad, 
                                                 mode=mode,
                                                 seed=seed)

        gen_chain = (self.raw_gen_test() |
                     add_bin_channel(obj_value=255) |
                     add_boundaries_channel(bnd_value=255) |
                     subtract_boundaries_from_objects() |
                     augmentator.flow(augs=augs, ensure_gt_binary=True) |
                     rescaler(trim_method=trim_method, out_dim=out_dim) |
                     normalizer() |
                     form_tensors() |
                     batch_generator(batch_size=4) |
                     form_dcan_input())

        input_imgs, input_gts = next(gen_chain)

        nptest.assert_array_equal(input_imgs.shape, (4, 256, 256, 1))
        self.assertTrue(np.max(input_imgs) == 1.0)
        self.assertTrue(np.min(input_imgs) == 0.0)
        self.assertTrue(input_imgs.dtype == np.float64)

        self.assertTrue(isinstance(input_gts, list))
        nptest.assert_array_equal(input_gts[0].shape, (4, 256, 256, 1))
        nptest.assert_array_equal(input_gts[1].shape, (4, 256, 256, 1))

        self.assertTrue(input_gts[0].dtype == np.uint8)
        self.assertTrue(input_gts[1].dtype == np.uint8)

        # Check objects values' range.
        for gt in input_gts[0]:
            self.assertTrue(np.max(gt) == 1)
            self.assertTrue(np.min(gt) == 0)

    def test_obj_bnd_channels_binary_when_augs_applied(self):
        augs = 1
        seed = 43
        mode = 'constant'
        pad = None

        augmentator = augmentations.AugGenerator(pad=pad,
                                                 mode=mode,
                                                 seed=seed)

        gen_chain = (self.raw_gen_test() |
                     add_bin_channel(obj_value=255) |
                     add_boundaries_channel(bnd_value=255) |
                     subtract_boundaries_from_objects() |
                     augmentator.flow(augs=augs, ensure_gt_binary=True))

        for _, aug_gt in gen_chain:
            self.assertTrue(len(np.unique(aug_gt[..., 1])) == 2)
            self.assertTrue(np.max(aug_gt[..., 1]) == 255)
            self.assertTrue(np.min(aug_gt[..., 1]) == 0)

            self.assertTrue(len(np.unique(aug_gt[..., 2])) == 2)
            self.assertTrue(np.max(aug_gt[..., 2]) == 255)
            self.assertTrue(np.min(aug_gt[..., 2]) == 0)

    def test_single_image_from_batch_when_single_image_generator(self):
        gen_chain = (self.raw_gen_test() |
                     form_tensors() |
                     batch_generator(batch_size=2) |
                     single_image_generator(multiple_gt_outputs=False))

        counter = 0
        for img, gt in gen_chain:
            self.assertEqual(img.shape, (100, 100, 1))
            self.assertEqual(gt.shape, (100, 100, 1))
            counter += 1

        self.assertEqual(counter, 4)

    def test_single_image_from_DCAN_batch_when_single_image_generator(self):
        gen_chain = (self.raw_gen_test() |
                     add_bin_channel(obj_value=255) |
                     add_boundaries_channel(bnd_value=255) |
                     form_tensors() |
                     batch_generator(batch_size=2) |
                     form_dcan_input() |
                     single_image_generator(multiple_gt_outputs=True))

        counter = 0
        for img, gt in gen_chain:
            self.assertEqual(img.shape, (1, 100, 100, 1))
            self.assertTrue(isinstance(gt, list))
            self.assertEqual(gt[0].shape, (1, 100, 100, 1))
            self.assertEqual(gt[1].shape, (1, 100, 100, 1))
            counter += 1

        self.assertEqual(counter, 4)

    def test_single_img_identical_to_batched_data_in_single_img_generator(self):
        gen_chain_dcan = (self.raw_gen_test() |
                          add_bin_channel(obj_value=255) |
                          add_boundaries_channel(bnd_value=255) |
                          form_tensors() |
                          batch_generator(batch_size=2) |
                          form_dcan_input())

        gen_chain_dcan_copy, _ = tee(gen_chain_dcan)

        gen_chain_single = (gen_chain_dcan_copy |
                            single_image_generator(multiple_gt_outputs=True))

        for img_batch, gt_batch in gen_chain_dcan_copy:
            for n in range(img_batch.shape[0]):
                img, gt = next(gen_chain_single)
                nptest.assert_array_equal(img_batch[np.newaxis, n], img)
                for out_idx, output in enumerate(gt):
                    nptest.assert_array_equal(gt_batch[out_idx][np.newaxis, n], 
                                              output)

    def tearDown(self):
        pass
