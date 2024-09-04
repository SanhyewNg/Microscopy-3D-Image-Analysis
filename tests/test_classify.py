import os
import shutil
import tempfile
import pytest
import unittest
from unittest.mock import patch

import attrdict
import imageio
import numpy as np
import numpy.testing as nptest
import skimage.filters

import clb.classify.classify as classify
import clb.dataprep.readers as readers
import clb.dataprep.tif.tif_readers as tif_readers
import clb.classify.predictors
from clb.classify.feature_extractor import extract_all_features
from clb.classify.extractors import extract_channels
from clb.classify.instance_matching import cell_level_classes
from clb.classify.train import (main as run_classify_train,
                                parse_arguments as parse_train)
from tests.utils import MockHitException, TestCase


@pytest.mark.classification
@pytest.mark.integration
class TestClassify(unittest.TestCase):
    tmp_dir = "test_classify_tmp"

    def setUp(self):
        self.addCleanup(self.tearDownAlways)

        np.random.seed(10)

        os.makedirs(os.path.join(self.tmp_dir, "input"), exist_ok=True)
        os.makedirs(os.path.join(self.tmp_dir, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.tmp_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.tmp_dir, "classes"), exist_ok=True)

        self.input_file = os.path.join(self.tmp_dir, "input", "file_1.tif")
        self.annotations_file = os.path.join(self.tmp_dir, "annotations",
                                             "annot_Ki67_1.tif")
        self.segmentation_dir = os.path.join(self.tmp_dir, 'labels')
        self.segmentation_name = 'file_1_segmented.tif'
        self.segmentation_file = os.path.join(self.segmentation_dir,
                                              self.segmentation_name)
        self.classification = os.path.join(self.tmp_dir, "classes",
                                           "class_Ki67_1.tif")

        input = (np.random.random((5, 100, 100, 3)) * 200)
        input = skimage.filters.gaussian(input, 0.8).astype(dtype=np.uint8)
        annotations = np.zeros((5, 100, 100), dtype=np.uint8)
        labels = np.zeros((5, 100, 100), dtype=np.uint8)

        for k in range(1, 50):
            pos_z = np.random.randint(0, 5, dtype=np.uint8),
            pos_yx = np.random.randint(0, 100,
                                       dtype=np.uint8), np.random.randint(0,
                                                                          100,
                                                                          dtype=np.uint8),
            annotations[pos_z][pos_yx[0] - 4:pos_yx[0] + 4,
            pos_yx[1] - 4:pos_yx[1] + 4] = 1
            labels[pos_z][pos_yx[0] - 3:pos_yx[0] + 3,
            pos_yx[1] - 5:pos_yx[1] + 5] = k
            input[pos_z][pos_yx[0] - 4:pos_yx[0] + 4,
            pos_yx[1] - 4:pos_yx[1] + 4] += 50

        self.input_file_data = input
        self.annotations_file_data = annotations
        self.segmentation_file_data = labels
        imageio.mimwrite(self.input_file, list(self.input_file_data))
        imageio.mimwrite(self.annotations_file, self.annotations_file_data)
        imageio.mimwrite(self.segmentation_file, self.segmentation_file_data)

    @patch('clb.classify.predictors.load_predictor')
    @patch('clb.classify.predictors.FeaturePredictor.predict_discrete')
    @patch('clb.classify.train.train_model')
    def test_same_features(self, train_model_mock, predict_mock,
                           load_model_mock):
        load_model_mock.return_value = clb.classify.predictors.FeaturePredictor()
        features_predict = []
        features_train = []

        def predict_mock_call(features, crop_data, *args):
            features_predict.append(features)
            raise MockHitException("Predict mock was hit.")

        def train_mock_call(model, x_y, *args):
            features_train.append(x_y[0])
            raise MockHitException("Train mock was hit.")

        predict_mock.side_effect = predict_mock_call
        train_model_mock.side_effect = train_mock_call

        test_model = os.path.join(self.tmp_dir, "class_model.pkl")
        final_test_model_path = os.path.join(self.tmp_dir,
                                             "class_model_class_Ki67.pkl")

        def assert_same_with_params(features_type, channels, resize):
            features_predict.clear()
            features_train.clear()
            with self.assertRaises(MockHitException):
                params = [
                    "--class_name", "Ki67",
                    "--model", test_model,
                    "--instance_model", "dummy",
                    # should use provided labels instead
                    "--annotated_input", os.path.dirname(self.input_file),
                    "--annotated_gt", os.path.dirname(self.annotations_file),
                    "--labels", os.path.dirname(self.segmentation_file),
                    "--est_class_weight", "balanced",
                    "--est_min_samples_leaf", "2",
                    "--features_type", features_type,
                    "--n_estimators", "100",
                    "--channels", channels,
                    "--no_voxel_resize"
                ]
                if not resize:
                    params += ["--no_voxel_resize"]

                args = parse_train(params)
                run_classify_train(args)

            with self.assertRaises(MockHitException):
                classify.classify(input=self.input_file,
                                  outputs=[self.classification],
                                  model=final_test_model_path,
                                  labels=self.segmentation_file,
                                  features_type=args.features_type,
                                  channels=args.channels,
                                  voxel_resize=resize)

            self.assertEqual(1, len(features_predict))
            self.assertEqual(len(features_predict), len(features_train))
            nptest.assert_array_equal(features_predict[0].columns,
                                      features_train[0].columns)
            nptest.assert_array_almost_equal(features_predict[0].sort_index(),
                                             features_train[0].sort_index(), 7)

        assert_same_with_params(features_type="perc_per_10", channels="1,0-clahe", resize=False)
        assert_same_with_params(features_type="complex", channels="1", resize=False)
        assert_same_with_params(features_type="complex+texture", channels="0-memb", resize=False)
        assert_same_with_params(features_type="all", channels="0-edges", resize=False)
        assert_same_with_params(features_type="all", channels="0-memb, 0-edges", resize=False)

    @patch('clb.classify.predictors.load_predictor')
    @patch('clb.classify.nn.predict_cube.predict_cube')
    def test_dl_predictor_input(self, predict_cube_mock, load_predictor_mock):
        load_predictor_mock.return_value = clb.classify.predictors.DLPredictor()
        test_model_path = os.path.join(self.tmp_dir, "vgg_model_class_CD8.pkl")
        model_passed = []
        model_path_passed = []
        input_volumes_passed = []

        def predict_cube_mock_call(model, model_path, input_volumes):
            model_passed.append(model)
            model_path_passed.append(model_path)
            input_volumes_passed.append(input_volumes)
            raise MockHitException("Predict mock was hit.")

        predict_cube_mock.side_effect = predict_cube_mock_call

        with self.assertRaises(MockHitException):
            classify.classify(input=self.input_file,
                              outputs=[self.classification],
                              model=test_model_path,
                              labels=self.segmentation_file,
                              features_type=None,
                              use_cubes=5.0,  # 5um so it is 10 pixels
                              channels="0,1")

        self.assertEqual(1, len(model_passed))
        input_volumes_passed = input_volumes_passed[0]
        real_objects_count = len(np.unique(self.segmentation_file_data)) - 1  # how many objects are there actually?
        self.assertEqual(real_objects_count, len(input_volumes_passed))
        self.assertEqual((10, 10, 10, 2), input_volumes_passed[0].shape)

    def test_same_area_features_matching(self):
        matching = cell_level_classes(self.segmentation_file_data,
                                      self.annotations_file_data,
                                      type="tissue")
        features = extract_all_features(self.input_file_data,
                                        self.segmentation_file_data, ['0'],
                                        "default")
        for k, v in matching.items():
            self.assertAlmostEqual(v['class_pixels'] / v['class_fraction'],
                                   features[k]['area'], 4)

    def tearDownAlways(self):
        shutil.rmtree(self.tmp_dir)


@pytest.mark.classification
@pytest.mark.integration
class TestClassifyTif(TestCase):
    def setUp(self):
        self.tif = 'tests/test_images/lif/series_0.tif'
        self.tif_labels = 'tests/test_images/lif/series_0_labels.tif'
        self.tif_classes_1 = 'tests/test_images/lif/series_0_classes_1.tif'
        self.tif_classes_2 = 'tests/test_images/lif/series_0_classes_2.tif'
        self.output_dir = tempfile.mkdtemp()

        def run(*args, **kwargs):
            outputs = kwargs['outputs']

            with readers.get_volume_reader(self.tif_labels) as labels:
                imageio.mimwrite(outputs[0], labels)

        patch_run = patch(target='clb.classify.classify.run.make_instance_segmentation',
                          new=run)
        patch_run.start()
        self.addCleanup(patch_run.stop)

        def get_features(input_volume, labels, channels, *args, **kwargs):
            channels = extract_channels(channels)
            input_channel = np.squeeze(input_volume[..., channels])
            labels = np.squeeze(labels)
            classes = np.logical_and(input_channel > 0, labels > 0)
            classes = np.where(classes, 255, 0).astype(np.uint8)

            return classes

        patch_get_features = patch(
            target='clb.classify.classify.get_features',
            new=get_features)
        patch_get_features.start()
        self.addCleanup(patch_get_features.stop)

        def predict(model, labels, features, cropped_cubes, discrete):
            return features

        patch_predict = patch('clb.classify.classify.overlay_predictions',
                              new=predict)
        patch_predict.start()
        self.addCleanup(patch_predict.stop)

    def tearDown(self):
        if os.path.isfile('classes.tif'):
            os.remove('classes.tif')

    def test_tif_output_with_existing_segmentation(self):
        output_path = os.path.join('classes.tif')
        classify.classify(input=self.tif,
                          outputs=[output_path],
                          model='dummy',
                          channels='1',
                          features_type='',
                          labels=self.tif_labels)

        self.assert_tif_equal_to_tif(output_path, self.tif_classes_1)

    def test_ims_append_output_without_existing_segmentation(self):
        output_path = os.path.join(self.output_dir, 'classes.ims')
        labels_path = os.path.join(self.output_dir, 'labels.tif')

        classify.classify(input=self.tif,
                          outputs=[output_path],
                          model='dummy',
                          channels='1',
                          features_type='',
                          labels=labels_path)
        classify.classify(input=self.tif,
                          outputs=[output_path],
                          model='dummy',
                          channels='2',
                          features_type='',
                          labels=labels_path)

        self.assert_tif_equal_to_tif(labels_path, self.tif_labels)
        self.assert_ims_equal_to_tifs(output_path,
                                      (self.tif, 0),
                                      self.tif_labels,
                                      (self.tif, 1),
                                      self.tif_classes_1,
                                      (self.tif, 2),
                                      self.tif_classes_2)


@pytest.mark.classification
@pytest.mark.integration
class TestClassifyLif(TestCase):
    def setUp(self):
        img_path = 'tests/test_images/lif'
        self.series_0 = os.path.join(img_path, 'series_0.tif')
        self.series_1 = os.path.join(img_path, 'series_1.tif')
        self.series_0_labels = os.path.join(img_path, 'series_0_labels.tif')
        self.series_1_labels = os.path.join(img_path, 'series_1_labels.tif')
        self.series_0_classes_1 = os.path.join(img_path,
                                               'series_0_classes_1.tif')
        self.series_0_classes_2 = os.path.join(img_path,
                                               'series_0_classes_2.tif')
        self.series_1_classes_1 = os.path.join(img_path,
                                               'series_1_classes_1.tif')
        self.series_1_classes_2 = os.path.join(img_path,
                                               'series_1_classes_2.tif')

        self.output_dir = tempfile.mkdtemp()

        def run(*args, **kwargs):
            outputs = kwargs['outputs']
            series = kwargs['series']

            if series == 0:
                path = self.series_0_labels
            else:
                path = self.series_1_labels

            with readers.get_volume_reader(path) as labels:
                imageio.mimwrite(outputs[0], labels)

        patch_run = patch(target='clb.classify.classify.run.make_instance_segmentation',
                          new=run)
        patch_run.start()
        self.addCleanup(patch_run.stop)

        def get_features(input_volume, labels, channels, *args, **kwargs):
            channels = extract_channels(channels)
            input_channel = np.squeeze(input_volume[:, channels])
            labels = np.squeeze(labels)
            classes = np.logical_and(input_channel > 0, labels > 0)
            classes = np.where(classes, 255, 0).astype(np.uint8)

            return classes

        patch_get_features = patch(
            target='clb.classify.classify.get_features',
            new=get_features)
        patch_get_features.start()
        self.addCleanup(patch_get_features.stop)

        def predict(model, labels, features, cropped_cubes, discrete):
            return features

        patch_predict = patch('clb.classify.classify.overlay_predictions',
                              new=predict)
        patch_predict.start()
        self.addCleanup(patch_predict.stop)

        def mock_get_reader(path, series=0):
            metadata = {
                'PhysicalSizeX': 0.5,
                'PhysicalSizeY': 0.5,
                'PhysicalSizeZ': 1.,
                'PhysicalSizeXUnit': 'um',
                'PhysicalSizeYUnit': 'um',
                'PhysicalSizeZUnit': 'um',
                'marker': 'dapi'
            }
            if path.endswith('.tif'):
                reader = tif_readers.TifReader(path)

            else:
                if series == 0:
                    reader = tif_readers.TifReader(self.series_0)
                    reader.metadata = {'Name': 'series_0'}
                else:
                    reader = tif_readers.TifReader(self.series_1)
                    reader.metadata = {'Name': 'series_1'}

                reader.dimensions['s'] = 2
                reader.metadata.update(metadata)

            return reader

        patch_get_volume_reader = patch(
            target='clb.classify.classify.readers._get_reader',
            new=mock_get_reader)
        patch_get_volume_reader.start()
        self.addCleanup(patch_get_volume_reader.stop)

    def test_tif_output_with_existing_segmentation(self):
        output_path = os.path.join(self.output_dir, 'classes.tif')
        classify.classify(input='input.lif',
                          outputs=[output_path],
                          model='dummy',
                          channels='1',
                          features_type='',
                          labels=self.series_1_labels,
                          series=1)

        self.assert_tif_equal_to_tif(output_path, self.series_1_classes_1)

    def test_tif_output_without_existing_segmentation(self):
        output_path = os.path.join(self.output_dir, 'classes.tif')
        labels_path = os.path.join(self.output_dir, 'labels.tif')
        classify.classify(input='input.lif',
                          outputs=[output_path],
                          model='dummy',
                          channels='2',
                          features_type='',
                          labels=labels_path,
                          series=0)

        self.assert_tif_equal_to_tif(labels_path, self.series_0_labels)
        self.assert_tif_equal_to_tif(output_path, self.series_0_classes_2)
