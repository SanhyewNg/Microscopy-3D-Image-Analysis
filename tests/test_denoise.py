import os
import shutil
import unittest
from unittest.mock import Mock, patch

import imageio
import numpy as np
import numpy.testing as nptest

import clb.denoising.denoise
import clb.denoising.train
from clb.dataprep.generators import preprocess
from clb.predict.predict3d import denoising_preprocess
from tests.utils import TestCase, MockHitException


class TestDenoise(TestCase):
    tmp_dir = 'temp_test_data'
    pregenerated_path = os.path.join('tests', 'test_images', 'denoising')
    train_path = os.path.join(pregenerated_path, 'train')
    denoise_model_path = 'models/denoising/model0.h5'

    input_path = os.path.join(tmp_dir, 'images')
    output_path = os.path.join(tmp_dir, 'denoised')

    def setUp(self):
        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

        self.input_image_path = os.path.join(self.pregenerated_path, 'input_0.tif')
        self.input_image_train = os.path.join(self.train_path, 'input_0.tif')
        self.denoised_path = os.path.join(self.output_path, "0.tif")
        self.expected_denoised_path = os.path.join(self.pregenerated_path, 'denoised_0.tif')

    @patch('clb.denoising.denoise.models.load_model')
    def test_denoise_process_with_identity(self, load_model_mock):
        # Mock it so it does not do anything.
        load_model_mock.return_value.predict = Mock(side_effect=lambda x: x)

        clb.denoising.denoise.denoise(self.input_image_path, self.denoised_path, self.denoise_model_path)

        load_model_mock.assert_called_with(self.denoise_model_path)
        self.assert_tif_equal_to_tif(self.input_image_path, self.denoised_path)

        # Without mock it can run actual model and result should be similar to the saved one.
        # TODO this actually hangs on Jenkins
        # self.assert_tif_equal_to_tif(self.expected_denoised_path, self.denoised_path, atol=2)

    @patch('clb.denoising.denoise.models.load_model')
    def test_same_preprocess_in_train_and_predict(self, load_model_mock):
        image_to_denoise = []

        def save_identify(x):
            image_to_denoise.append(x)
            return x

        load_model_mock.return_value.predict = Mock(side_effect=save_identify)

        training_generator = clb.denoising.train.make_training_pipeline(self.train_path, 1, shuffle=False)
        clb.denoising.denoise.denoise(self.input_image_train, self.denoised_path, self.denoise_model_path)

        self.assertEquals(5, len(image_to_denoise))  # number of z-slices
        nptest.assert_equal(image_to_denoise[0], next(training_generator)[0])

    @patch('clb.denoising.denoise.denoise_image')
    @patch('clb.denoising.denoise.models.load_model')
    def test_denoise_uses_denoise_image(self, load_model_mock, denoise_image_mock):
        self.set_stop_mock(denoise_image_mock)

        with self.assertRaises(MockHitException):
            clb.denoising.denoise.denoise(self.input_image_path, self.denoised_path, self.denoise_model_path)

        self.assert_called_once_including(denoise_image_mock,
                                          batch_size=1, patches_shape=None, patches_stride=None)

    @patch('clb.predict.predict3d.load_model_with_cache')
    @patch('clb.denoising.denoise.denoise_image')
    def test_segmentation_preprocess_uses_denoise_image(self, denoise_image_mock, load_model_mock):
        self.set_stop_mock(denoise_image_mock)

        input_slices = imageio.volread(self.input_image_path)
        with self.assertRaises(MockHitException):
            denoised = next(denoising_preprocess(input_slices))

        self.assert_called_once_including(denoise_image_mock,
                                          batch_size=1, patches_shape=None, patches_stride=None)

    @patch('clb.dataprep.generators.load_model_with_cache')
    @patch('clb.denoising.denoise.denoise_image')
    def test_denoising_generator_uses_denoise_image(self, denoise_image_mock, load_model_mock):
        generator_denoiser = preprocess(preprocessings=['denoising'])

        self.set_stop_mock(denoise_image_mock)

        input_slices = imageio.volread(self.input_image_path)
        with self.assertRaises(MockHitException):
            denoised = next([(input_slices[0], input_slices[1])] | generator_denoiser)

        self.assert_called_once_including(denoise_image_mock,
                                          batch_size=1, patches_shape=None, patches_stride=None)

    def tearDown(self):
        shutil.rmtree("temp_test_data")
