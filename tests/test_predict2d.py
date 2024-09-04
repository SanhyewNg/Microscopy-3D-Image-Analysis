import os
import shutil
import unittest
from unittest.mock import Mock, patch

import imageio
import numpy as np
import numpy.testing as nptest

import clb.predict.predict_tile
import clb.train.predict
import clb.train.train
from clb.predict.predict2d import segmentation2d_by_tiles, segmentation2d_dcan_tile


class TestPredict2d(unittest.TestCase):

    data_dir = 'temp_test_data'
    input_path = os.path.join(data_dir, 'images')
    labels_path = os.path.join(data_dir, 'labels')

    def setUp(self):
        # create temp folder
        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.labels_path, exist_ok=True)

        # create random input data with a square
        self.image = (np.random.random((200, 200)) * 128).astype(np.uint8)
        self.image[80:120, 80:120] = 255
        imageio.mimwrite(os.path.join(self.input_path, "0.tif"), [self.image, self.image])
        # create random label for a square
        self.label = np.zeros((200, 200), dtype=np.uint8)
        self.label[80:120, 80:120] = 255
        imageio.mimwrite(os.path.join(self.labels_path, "0.tif"), [self.label, self.label])

    class MockModel:
        def __init__(self):
            self.images = None

        def predict(self, images, **kwargs):
            self.images = images
            return None

        def predict_generator(self, generator, **kwargs):
            """Mock `model.predict_generator()` method from Keras `model` API.

            This method is called in `clb.train.predict()` function
            instead of `model.predict_generator()` one. The aim is to capture
            the data that enters the network (in this case in form of
            `generator` parameter), get first image of that generator (since
            we're working with batch size equal to one and no augs in this
            test) and store in in `self.images` attribute that will be
            compared in `test_same_for_method()` method.

            Args:
                generator: data generator that is used to generate tuples of
                           examples (img, gt) for the training.
            
            Returns:
                None
            """
            self.images, _ = next(generator)
            return None


    @staticmethod
    def get_predict_input(action, params=(), trim_method=None):
        # Replace load_model_with_cache with function that gives mock that
        # saves provided images.
        try:
            orig_load = clb.predict.predict_tile.load_model_with_cache
            mock_model = TestPredict2d.MockModel()
            clb.predict.predict_tile.load_model_with_cache = lambda x: mock_model
            clb.train.predict.load_model_with_cache = lambda x: mock_model

            param_dict = {
                'architecture': 'dcan',
                'channels': 1,
                'im_dim': 256,
                'test_data': 'temp_test_data',
                'trim_method': trim_method,
                'seed': 48,
                'model': ''
            }

            with patch('clb.train.predict.merge_cli_to_yaml',
                       new=Mock(return_value=param_dict)):
                action(*params)

        except TypeError as ex:
            if mock_model.images is None:
                # Exception before predict - something is broken
                raise
            else:
                # Exception after predict - it is just a mock
                pass
        finally:
            clb.predict.predict_tile.load_model_with_cache = orig_load
            clb.train.predict.load_model_with_cache = orig_load
        return mock_model.images

    def test_same_preprocessing_2d(self):
        def create_segment2d_with_dcan(trim_method):
            def segment2d_with_dcan():
                result = segmentation2d_by_tiles(self.image, 0,
                                                 lambda x: segmentation2d_dcan_tile(x,
                                                                                    trim_method=trim_method,
                                                                                    postprocess=False,
                                                                                    model_path='model.h5'), 200)

                return result
            return segment2d_with_dcan

        def prepare_main_parameters():
            params = ('', self.data_dir)
            return params

        def test_same_for_method(trim_method, precision=15):
            input_to_predict_segment2d = TestPredict2d.get_predict_input(
                create_segment2d_with_dcan(trim_method))

            # run main.py with that image in predict mode
            params = prepare_main_parameters()
            input_to_predict_main = TestPredict2d.get_predict_input(
                clb.train.predict.predict, params, trim_method)

            # compare images - same values same dtype
            nptest.assert_array_almost_equal(input_to_predict_segment2d,
                                             input_to_predict_main,
                                             precision)

            self.assertEqual(input_to_predict_segment2d.dtype,
                             input_to_predict_main.dtype)

            self.assertEqual(input_to_predict_segment2d.shape,
                             input_to_predict_main.shape)

        test_same_for_method('padding')
        test_same_for_method('reflect')
        test_same_for_method('resize', 2)

    def test_segmentation2d_by_tiles_grayscale(self):
        # create image that can be split into 2.5
        # let it be grayscale
        big_image = (np.random.random((100, 100)) * 120).astype(np.uint8)
        big_image[0:10, 0:20] = 243
        big_image[30:95, 33:79] = 222

        def threshold_test_segm(imgs):
            res = []
            for im in imgs:
                thresh = im / 250.0
                thresh[thresh < 0.3] = 0
                res.append(thresh)
            return res

        segmentation_joined = segmentation2d_by_tiles(big_image, pad_size=10, tile_size=40,
                                                      segmentation_tile=threshold_test_segm)
        self.assertEqual(big_image.shape, segmentation_joined.shape)

        on_all = threshold_test_segm([big_image])
        nptest.assert_array_equal(on_all, [segmentation_joined])

    def test_segmentation2d_by_tiles_colour(self):
        # create image that can be split into 2.5
        # let it be colour
        big_image = (np.random.random((100, 100, 3)) * 120).astype(np.uint8)
        big_image[0:10, 0:20] = 243
        big_image[30:95, 33:79] = 222

        def threshold_test_segm(imgs):
            res = []
            for im in imgs:
                thresh = (im / 250.0)[::,::,0]
                thresh[thresh < 0.3] = 0
                res.append(thresh)
            return res

        segmentation_joined = segmentation2d_by_tiles(big_image, pad_size=10, tile_size=40,
                                                      segmentation_tile=threshold_test_segm)
        self.assertEqual(big_image.shape[:2], segmentation_joined.shape)

        on_all = threshold_test_segm([big_image])
        nptest.assert_array_equal(on_all, [segmentation_joined])

    def tearDown(self):
        shutil.rmtree("temp_test_data")
