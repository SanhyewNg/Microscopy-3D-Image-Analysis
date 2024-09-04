import argparse
import os
import pytest
import shutil
import tempfile
import unittest

import numpy as np
import numpy.testing as nptest
import skimage
import yaml

from clb.dataprep.generators import (add_bin_channel, add_boundaries_channel,
                                     batch_generator, form_dcan_input,
                                     form_tensors)
from clb.dataprep.utils import reduce_to_max_channel
from clb.train.utils import get_images_for_tensorboard
from clb.utils import (bbox, has_gaps, normalize_channels_volume,
                       replace_values, replace_values_in_slices)
from clb.volume_slicer import split_volume
from clb.yaml_utils import (load_yaml_args, merge_yaml_to_cli,
                            save_args_to_yaml, load_args, save_args)
from vendor.genny.genny.wrappers import obj_gen_wrapper


class TestUtils(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

    @obj_gen_wrapper
    def raw_gen_test(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 1

        values = [(img, gt), (img, gt), (img, gt), (img, gt)]

        for tup in values:
            yield tup

    @pytest.mark.preprocessing
    def test_image_kept_unchanged_within_slice_in_spatial_context(self):
        img_stack = np.random.randint(0, 256, (5, 200, 200), dtype=np.uint8)
        gt_stack = np.random.randint(0, 256, (5, 200, 200), dtype=np.uint8)
        slice_thickness = 3
        img_stack_sliced, _ = split_volume(img_stack, gt_stack,
                                           slice_thickness,
                                           spatial_context=True)

        single_slice = img_stack_sliced[0]

        input_image = np.squeeze(img_stack[0, :, :])
        output_image = np.squeeze(single_slice[:, :, 0])
        nptest.assert_equal(input_image, output_image)
        input_image = np.squeeze(img_stack[1, :, :])
        output_image = np.squeeze(single_slice[:, :, 1])
        nptest.assert_equal(input_image, output_image)
        input_image = np.squeeze(img_stack[2, :, :])
        output_image = np.squeeze(single_slice[:, :, 2])
        nptest.assert_equal(input_image, output_image)

        single_slice = img_stack_sliced[1]

        input_image = np.squeeze(img_stack[1, :, :])
        output_image = np.squeeze(single_slice[:, :, 0])
        nptest.assert_equal(input_image, output_image)
        input_image = np.squeeze(img_stack[2, :, :])
        output_image = np.squeeze(single_slice[:, :, 1])
        nptest.assert_equal(input_image, output_image)
        input_image = np.squeeze(img_stack[3, :, :])
        output_image = np.squeeze(single_slice[:, :, 2])
        nptest.assert_equal(input_image, output_image)

    @pytest.mark.preprocessing
    def test_replace_values(self):
        image1 = np.zeros((20, 20), dtype=np.uint8)
        image1[10] = 1
        image1[13] = 2
        image1[14] = 3
        replace_values(image1, {1: 5, 2: 6}, return_copy=False)
        self.assertEqual(5, image1[10][2])
        self.assertEqual(6, image1[13][2])
        self.assertEqual(3, image1[14][2])

    @pytest.mark.preprocessing
    def test_replace_values_from_to_overlaping(self):
        image1 = np.zeros((3, 20, 20), dtype=np.uint8)
        image1[1][2] = 1
        image1[2][0:3, 2] = 4
        image1[2][0][0] = 8
        image1[2][4:6] = 9

        replace_values(image1, {2: 5, 8: 9, 9: 8}, return_copy=False)
        self.assertEqual(0, image1[0, 0, 0])
        self.assertTrue(np.all(image1[1][2] == 1))
        self.assertTrue(np.all(image1[2][0:3, 2] == 4))
        self.assertEqual(9, image1[2, 0, 0])
        self.assertTrue(np.all(image1[2][4:6] == 8))

    @pytest.mark.preprocessing
    def test_replace_values_in_slices(self):
        image1 = np.zeros((20, 20), dtype=np.uint8)
        image1[10] = 1
        image1[13] = 2
        image1[14] = 3
        image2 = np.zeros((20, 20), dtype=np.uint8)
        image2[13] = 1
        image2[12] = 3
        image3 = np.zeros((20, 20), dtype=np.uint8)
        image3[10] = 1
        image3[11] = 2
        replace_values_in_slices([image1, image2, image3],
                                 {1: 10, 2: 20, 3: 30, 4: 40})

        self.assertEqual(10, image1[10][2])
        self.assertEqual(20, image1[13][2])
        self.assertEqual(30, image1[14][2])
        self.assertEqual(10, image2[13][2])
        self.assertEqual(30, image2[12][2])
        self.assertEqual(10, image3[10][2])

        # does not see image3[11] value at it is not expected
        self.assertEqual(2, image3[11][2])

    @pytest.mark.preprocessing
    def test_normalize_channels_image_single_channel(self):
        image = np.random.random((200, 200))

        unchanged = reduce_to_max_channel(image)
        nptest.assert_equal(image, unchanged)
        self.assertEqual(True, image is unchanged)

    @pytest.mark.preprocessing
    def test_normalize_channels_image_multi_channel(self):
        image = np.random.random((200, 200))
        rgb = np.zeros((200, 200, 3))
        rgb[::, ::, 1] = image

        normalized = reduce_to_max_channel(rgb)
        nptest.assert_equal(image, normalized)
        self.assertEqual(False, image is normalized)

        rgb2 = np.zeros((2, 200, 200))
        rgb2[1, ::, ::] = image

        normalized = reduce_to_max_channel(rgb2)
        nptest.assert_equal(image, normalized)
        self.assertEqual(False, image is normalized)

    @pytest.mark.preprocessing
    def test_normalize_channels_volume_single_channel(self):
        volume = np.random.random((2, 200, 200))
        unchanged = normalize_channels_volume(volume)
        nptest.assert_equal(volume, unchanged)
        self.assertEqual(True, volume is unchanged)

    @pytest.mark.preprocessing
    def test_normalize_channels_volume_multi_channel(self):
        volume = np.random.random((2, 200, 200))
        rgb = np.zeros((2, 200, 200, 3))
        rgb[::, ::, ::, 1] = volume

        normalized = normalize_channels_volume(rgb)
        nptest.assert_equal(volume, normalized)
        self.assertEqual(False, volume is normalized)

    @pytest.mark.io
    def test_yaml_save_load(self):
        some_file = 'data'
        some_data = {'bikes': 10, 'stopping': 'fast'}
        try:
            save_args_to_yaml(some_file, some_data)
            loaded_data = load_yaml_args(some_file + '.yaml')

            self.assertEqual(some_data, loaded_data)

            some_new_data = argparse.Namespace(**some_data)
            some_new_data.light = False
            some_new_data.bikes = 12
            some_new_data.stopping = 'slow'
            merged_data = merge_yaml_to_cli(some_file + '.yaml', vars(some_new_data), ['stopping'])

            self.assertEqual(False, merged_data.light)
            self.assertEqual(12, merged_data.bikes)
            self.assertEqual('fast', merged_data.stopping)
        finally:
            os.remove(some_file + '.yaml')

    @pytest.mark.preprocessing
    def test_relabel_unique(self):
        volume = np.zeros((2, 20, 20), dtype=np.uint16)
        volume[0][2:8, 2:8] = 2
        volume[0][0:3, 0:3] = 1
        volume[0][0:3, 7:10] = 1
        volume[0][5:8, 3:6] = 1

        volume[1][0:12, 4:11] = 1

        labels_unique = skimage.measure.label(volume)

        self.assertNotEqual(labels_unique[0][0, 0], labels_unique[0][0, 8])
        self.assertEqual(labels_unique[0][6, 4], labels_unique[0][0, 8])
        self.assertEqual(labels_unique[0][6, 4], labels_unique[1][6, 9])

    @pytest.mark.preprocessing
    def test_valid_tuple_returned_when_get_images_for_tensorboard(self):
        gen = (
                self.raw_gen_test() |
                add_bin_channel() |
                add_boundaries_channel() |
                form_tensors() |
                batch_generator(batch_size=2) |
                form_dcan_input()
        )

        num_imgs = 3

        imgs, gts = get_images_for_tensorboard(batch_gen=gen,
                                               num_imgs=num_imgs,
                                               architecture='dcan')

        self.assertEqual(imgs.shape, (3, 100, 100, 1))
        self.assertEqual(type(gts), list)
        self.assertEqual(len(gts), 2)
        self.assertEqual(gts[0].shape, (3, 100, 100, 1))
        self.assertEqual(gts[1].shape, (3, 100, 100, 1))

    @pytest.mark.preprocessing
    def test_bbox_1d(self):
        array_1 = np.array([1, 0, 0, 2], dtype=np.uint8)
        array_2 = np.array([0, 1, 0, 0, 2, 0, 0], dtype=np.uint8)

        box = bbox(array_1, 0)
        self.assertEqual((0, 3), box)

        box = bbox(array_2, 0)
        self.assertEqual((1, 4), box)

    @pytest.mark.preprocessing
    def test_bbox_2d(self):
        array_1 = np.array([[1, 0, 0, 2],
                            [0, 1, 2, 0]], dtype=np.uint8)

        array_2 = np.array([[0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 0, 0]], dtype=np.uint8)

        box = bbox(array_1, 0)
        self.assertEqual((0, 1), box)
        box = bbox(array_1, 1)
        self.assertEqual((0, 3), box)
        box = bbox(array_1, [0, 1])
        self.assertEqual([(0, 1), (0, 3)], box)

        box = bbox(array_2, 0)
        self.assertEqual((1, 2), box)
        box = bbox(array_2, 1)
        self.assertEqual((1, 1), box)

    @pytest.mark.preprocessing
    def test_bbox_3d(self):
        array_1 = np.array([[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],

                            [[0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [0, 0, 0, 0]],

                            [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]
                            ], dtype=np.uint8)

        box = bbox(array_1, [0, 1, 2])
        self.assertEqual([(1, 1), (0, 1), (2, 3)], box)

        box = bbox(array_1, [0])
        self.assertEqual((1, 1), box)

    @pytest.mark.preprocessing
    def test_bbox_empty(self):
        array_1 = np.array([[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],

                            [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]
                            ], dtype=np.uint8)

        box = bbox(array_1, [0, 1, 2])
        self.assertEqual(None, box)

    @pytest.mark.preprocessing
    def test_has_gaps(self):
        array = np.array([[[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]],

                          [[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]],

                          [[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]]
                          ], dtype=np.uint8)

        self.assertEqual(False, has_gaps(array))

        array[0, 0, 0] = 2
        self.assertEqual(False, has_gaps(array))

        array[2, 2, 1] = 4
        self.assertEqual(False, has_gaps(array))

        array[1] = 0
        self.assertEqual(True, has_gaps(array))

        array[2] = 0
        self.assertEqual(False, has_gaps(array))

    def tearDown(self):
        pass


@pytest.mark.io
class TestLoadArgs(unittest.TestCase):
    def setUp(self):
        results_dir = tempfile.mkdtemp()
        self.args_path = os.path.join(results_dir, 'args.yaml')
        self.addCleanup(shutil.rmtree, results_dir)
        self.arguments = {'a': 1, 'b': 2}
        with open(self.args_path, 'w') as f:
            yaml.dump(self.arguments, f)

    def test_reading_all_only_keyword_args(self):
        calls = []

        @load_args(arg_name='args_load_path')
        def f(*, a, b):
            calls.append({'a': a, 'b': b})

        f(args_load_path=self.args_path)

        self.assertEqual(calls, [self.arguments])

    def test_reading_one_only_keyword_args(self):
        calls = []

        @load_args(arg_name='args_load_path')
        def f(*, a, b):
            calls.append({'a': a, 'b': b})

        f(a=2, args_load_path=self.args_path)

        self.assertEqual(calls, [{'a': 2, 'b': 2}])

    def test_reading_all_with_default_args(self):
        calls = []

        @load_args(arg_name='args_load_path')
        def f(*, a, b=3):
            calls.append({'a': a, 'b': b})

        f(args_load_path=self.args_path)

        self.assertEqual(calls, [self.arguments])

    def test_without_reading(self):
        calls = []

        @load_args(arg_name='args_load_path')
        def f(*, a, b=3):
            calls.append({'a': a, 'b': b})

        f(a=2)

        self.assertEqual(calls, [{'a': 2, 'b': 3}])


@pytest.mark.io
class TestSaveArgs(unittest.TestCase):
    def setUp(self):
        results_dir = tempfile.mkdtemp()
        self.args_path = os.path.join(results_dir, 'args.yaml')
        self.addCleanup(shutil.rmtree, results_dir)

    def test_saving_all_arguments(self):
        @save_args(arg_name='args_save_path')
        def test_function(*, a, b):
            pass

        test_function(a=1, b=2, args_save_path=self.args_path)
        loaded_args = load_yaml_args(self.args_path)

        self.assertEqual(loaded_args, {'a': 1, 'b': 2})

    def test_saving_not_all_arguments(self):
        @save_args(arg_name='args_save_path')
        def test_function(*, a, b):
            pass

        test_function(a=1, b=2, args_save_path=self.args_path)
        loaded_args = load_yaml_args(self.args_path)

        self.assertEqual(loaded_args, {'a': 1, 'b': 2})


@pytest.mark.io
class TestLoadAndSaveArgs(unittest.TestCase):
    def setUp(self):
        results_dir = tempfile.mkdtemp()
        self.args_load_path = os.path.join(results_dir, 'args.yaml')
        self.addCleanup(shutil.rmtree, results_dir)
        self.arguments = {'a': 1, 'b': 2}
        with open(self.args_load_path, 'w') as f:
            yaml.dump(self.arguments, f)

        self.args_save_path = os.path.join(results_dir, 'saved_args.yaml')

    def test_saving_all_arguments(self):
        @load_args(arg_name='args_load_path')
        @save_args(arg_name='args_save_path')
        def test_function(*, a, b):
            pass

        test_function(args_save_path=self.args_save_path,
                      args_load_path=self.args_load_path)
        loaded_args = load_yaml_args(self.args_save_path)

        self.assertEqual(loaded_args, {'a': 1, 'b': 2})
