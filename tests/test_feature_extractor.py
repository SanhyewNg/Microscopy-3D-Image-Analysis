import pytest
import unittest

import numpy as np
import numpy.testing as nptest

import clb.classify.feature_extractor as feature_extractor
import clb.classify.extractors as extractors
import clb.classify.utils as utils

@pytest.mark.statistics
class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

    def filter_features_to_keys(self, features_data, keys_to_keep):
        res = {}
        for i in features_data.keys():
            res[i] = {p_k: p_v for p_k, p_v in features_data[i].items() if p_k in keys_to_keep}
        return res

    def generate_intensity_labels(self):
        intensity = np.random.random((3, 50, 50, 2))

        labels = np.zeros((3, 50, 50), dtype=np.uint8)
        labels[0, 3:10, 3:10] = 1
        labels[0, 12:16, 12:16] = 2
        labels[1, 5:10, 2:9] = 3
        labels[2, 5:8, 2:5] = 3
        return intensity, labels

    def generate_label_cases(self):
        labels = np.zeros((3, 50, 50), dtype=np.uint8)
        # make cube
        labels[0:3, 3:5, 3:5] = 2
        # make long stick
        labels[1, 10:15, 5] = 4
        labels[2, 15:20, 6] = 4
        # make 2d star
        labels[1, 25, 20:31] = 5
        labels[1, 20:31, 25] = 5
        return labels

    def test_extract_cells_features_types(self):
        intensity = np.random.random((3, 50, 50))

        labels = np.zeros((3, 50, 50), dtype=np.uint8)
        labels[0, 3:10, 3:10] = 1
        labels[0, 12:16, 12:16] = 2
        labels[1, 5:10, 2:9] = 3
        labels[2, 5:8, 2:5] = 3
        labels[2, 1:6, 10:15] = 4
        labels[1, 1:3, 1:3] = 7

        features = feature_extractor.extract_cells_features(intensity, labels, "default")
        self.assertEqual(5, len(features))
        self.assertIn("mean_intensity", features[1].keys())
        self.assertIn("median_intensity", features[1].keys())
        self.assertIn("perc_75_intensity", features[1].keys())

        features_complex = feature_extractor.extract_cells_features(intensity, labels, "complex+texture")
        self.assertIn("mean_intensity", features_complex[1].keys())
        self.assertNotIn("median_intensity", features_complex[1].keys())
        self.assertNotIn("perc_75_intensity", features_complex[1].keys())
        self.assertIn("perc_10_intensity", features_complex[1].keys())
        self.assertIn("perc_50_intensity", features_complex[1].keys())
        self.assertIn("perc_90_intensity", features_complex[1].keys())
        self.assertIn("haralick_mean_AngularSecondMoment", features_complex[1].keys())
        self.assertIn("haralick_peak2peak_AngularSecondMoment", features_complex[1].keys())
        self.assertIn("mad_intensity", features_complex[1].keys())

        for k in features_complex:
            self.assertEqual(features[k]['median_intensity'], features[k]['perc_50_intensity'])

        features_complex_no_texture = feature_extractor.extract_cells_features(intensity, labels, "complex")
        self.assertIn("mean_intensity", features_complex_no_texture[1].keys())
        self.assertNotIn("perc_75_intensity", features_complex_no_texture[1].keys())
        self.assertIn("perc_10_intensity", features_complex_no_texture[1].keys())
        self.assertNotIn("haralick_mean_AngularSecondMoment", features_complex_no_texture[1].keys())
        self.assertNotIn("haralick_peak2peak_AngularSecondMoment", features_complex_no_texture[1].keys())
        self.assertIn("mad_intensity", features_complex_no_texture[1].keys())

    def test_extract_cells_for_chosen_cells(self):
        intensity = np.random.random((3, 50, 50))

        labels = np.zeros((3, 50, 50), dtype=np.uint8)
        labels[0, 3:10, 3:10] = 1
        labels[0, 12:16, 12:16] = 2
        labels[1, 5:10, 2:9] = 3
        labels[2, 5:8, 2:5] = 3
        labels[2, 1:6, 10:15] = 4
        labels[1, 1:3, 1:3] = 7

        features = feature_extractor.extract_all_features(intensity, labels)
        self.assertEqual(5, len(features))

        selected_features = feature_extractor.extract_all_features(intensity, labels, only_for_labels=[1, 3])
        self.assertEqual(2, len(selected_features))
        self.assertEqual(features[1], selected_features[1])
        self.assertEqual(features[3], selected_features[3])

    def test_update_with_prefix(self):
        params = {"id": 1, "area": 10}
        size_A = {"id": 1, "area": 20}

        utils.update_with_prefix(params, size_A, "A")
        self.assertEqual({"id": 1, "area": 10, "A_area": 20}, params)

        utils.update_with_prefix(params, size_A, "B")
        self.assertEqual({"id": 1, "area": 10, "A_area": 20, "B_area": 20}, params)

    def test_join_data_with_prefix(self):
        params = {}
        first_channel_data = {1: {"id": 1, "area": 10}, 2: {"id": 2, "area": 20}}
        other_channel_data = {2: {"id": 2, "area": 15}, 1: {"id": 1, "area": 5}}

        utils.add_data_with_prefix(params, first_channel_data, "")
        utils.add_data_with_prefix(params, other_channel_data, "C")

        self.assertEqual({"id": 1, "area": 10, "C_area": 5}, params[1])
        self.assertEqual({"id": 2, "area": 20, "C_area": 15}, params[2])

        one_cell_data = {2: {"id": 2, "area": 7}}
        with self.assertRaises(AssertionError):
            utils.add_data_with_prefix(params, one_cell_data, "D")

    def test_join_data_with_prefix_mismatch(self):
        params = {1: {"id": 1, "area": 10}, 2: {"id": 2, "area": 20}}
        other_channel_data = {2: {"id": 2, "area": 15}, 1: {"id": 1, "area": 5}}

        utils.add_data_with_prefix(params, other_channel_data, "C")

        self.assertEqual({"id": 1, "area": 10, "C_area": 5}, params[1])
        self.assertEqual({"id": 2, "area": 20, "C_area": 15}, params[2])

    def test_extract_all_features_multichannel(self):
        intensity = np.random.random((3, 50, 50, 2))

        labels = np.zeros((3, 50, 50), dtype=np.uint8)
        labels[0, 3:10, 3:10] = 1
        labels[0, 12:16, 12:16] = 2
        labels[1, 5:10, 2:9] = 3
        labels[2, 5:8, 2:5] = 3

        features_0 = feature_extractor.extract_all_features(intensity, labels, ["0"], "default")[3]
        features_1 = feature_extractor.extract_all_features(intensity, labels, ["1"], "default")[3]

        features_0_1 = feature_extractor.extract_all_features(intensity, labels, ["0", "1"], "default")[3]

        self.assertEqual(features_0['id'], features_0_1['id'])
        self.assertEqual(False, '1_id' in features_0_1)
        self.assertEqual(features_0['0_mean_intensity'], features_0_1['0_mean_intensity'])
        self.assertEqual(features_1['1_mean_intensity'], features_0_1['1_mean_intensity'])

    def test_extract_clahe_features(self):
        intensity, labels = self.generate_intensity_labels()

        features_clahe = feature_extractor.extract_all_features(intensity, labels, ["0", "0-clahe"], "default")[3]
        self.assertTrue('0_mean_intensity' in features_clahe.keys())
        self.assertTrue('1_mean_intensity' not in features_clahe.keys())
        self.assertTrue('0-clahe_mean_intensity' in features_clahe.keys())

    def test_extract_all_features_unsupported_preprocessing(self):
        intensity, labels = self.generate_intensity_labels()

        self.assertEqual(True, any(feature_extractor.extract_all_features(intensity, labels, ["0"], "default")))
        with self.assertRaises(KeyError):
            feature_extractor.extract_all_features(intensity, labels, ["0-hipero"], "default")

    def test_get_feature_columns(self):
        sample_columns = ['1_blabla', 'id', 'class', 'pos_x', 'solidity', '0_mean_intensity', '1_mean_intensity',
                          '1_mad_intensity',
                          '1_haralick_00_SumEntropy', '1_moment_normalized_1_2_1']
        default_feature_columns = feature_extractor.get_feature_columns(sample_columns, 'default')

        self.assertEqual(['0_mean_intensity', '1_mean_intensity'], default_feature_columns)
        with self.assertRaises(Exception):
            feature_extractor.get_feature_columns(sample_columns, 'my_features')

    def test_get_feature_columns_complex(self):
        sample_columns = ['1_blabla', 'id', 'class', 'pos_x', 'solidity', '0_mean_intensity', '1_mean_intensity',
                          '1_mad_intensity',
                          '1_haralick_mean_SumEntropy', '1_moment_normalized_1_2_1']

        complex_feature_columns = feature_extractor.get_feature_columns(sample_columns, 'complex+texture')
        self.assertEqual(['0_mean_intensity', '1_haralick_mean_SumEntropy', '1_mad_intensity', '1_mean_intensity',
                          '1_moment_normalized_1_2_1', 'solidity'], complex_feature_columns)

        complex_1_feature_columns = feature_extractor.get_feature_columns(sample_columns, 'complex+texture', ["1"])
        self.assertEqual(['1_haralick_mean_SumEntropy', '1_mad_intensity', '1_mean_intensity',
                          '1_moment_normalized_1_2_1', 'solidity'], complex_1_feature_columns)

        complex_no_texture_1_feature_columns = feature_extractor.get_feature_columns(sample_columns, 'complex', ["1"])
        self.assertEqual(['1_mad_intensity', '1_mean_intensity',
                          '1_moment_normalized_1_2_1', 'solidity'], complex_no_texture_1_feature_columns)

        complex_0_feature_columns = feature_extractor.get_feature_columns(sample_columns, 'complex', ["0"])
        self.assertEqual(['0_mean_intensity', 'solidity'], complex_0_feature_columns)

    def test_get_feature_columns_chosen_preprocess(self):
        sample_columns = ['1_blabla', 'id', 'class', 'pos_x', '0_mean_intensity', '1_mean_intensity',
                          '1-clahe_mean_intensity']

        default_feature_columns = feature_extractor.get_feature_columns(sample_columns, 'default', ["0", "1"])
        self.assertEqual(['0_mean_intensity', '1_mean_intensity'], default_feature_columns)

        default_feature_columns_1 = feature_extractor.get_feature_columns(sample_columns, 'default', ["1"])
        self.assertEqual(['1_mean_intensity'], default_feature_columns_1)

        default_feature_columns_1 = feature_extractor.get_feature_columns(sample_columns, 'default', ["1-clahe"])
        self.assertEqual(['1-clahe_mean_intensity'], default_feature_columns_1)

        with self.assertRaises(Exception):
            feature_extractor.get_feature_columns(sample_columns, 'default', ["0", "0-clahe"])

    def test_extract_shape_features(self):
        labels = self.generate_label_cases()
        original_labels = labels.copy()

        shapes = feature_extractor.extract_shape_features(labels, 'complex')
        nptest.assert_array_equal(original_labels, labels)
        self.assertEqual([2, 4, 5], sorted(shapes.keys()))
        self.assertEqual(12, shapes[2]['area'])
        self.assertEqual(10, shapes[4]['area'])
        self.assertEqual(21, shapes[5]['area'])

        self.assertAlmostEqual(3.5, shapes[2]['pos_x'])
        self.assertAlmostEqual(5.5, shapes[4]['pos_x'])
        self.assertAlmostEqual(25, shapes[5]['pos_x'])

        self.assertEqual(3, shapes[2]['size_z'])
        self.assertEqual(2, shapes[4]['size_z'])
        self.assertEqual(1, shapes[5]['size_z'])

        self.assertAlmostEqual(1, shapes[2]['solidity'])
        self.assertAlmostEqual(1, shapes[4]['solidity'], 1)
        self.assertAlmostEqual(25 / (25 + 48.0), shapes[5]['solidity'], 2)

        self.assertAlmostEqual(0.67, shapes[2]['first_major_diff'], 2)
        self.assertAlmostEqual(0.99, shapes[4]['first_major_diff'], 2)
        self.assertAlmostEqual(0.0, shapes[5]['first_major_diff'], 2)

        self.assertAlmostEqual(0.0, shapes[2]['second_major_diff'], 2)
        self.assertAlmostEqual(0.12, shapes[4]['second_major_diff'], 2)
        self.assertAlmostEqual(0.71, shapes[5]['second_major_diff'], 2)

        shapes = feature_extractor.extract_shape_features(labels, 'perc_per_10')
        self.assertEqual([2, 4, 5], sorted(shapes.keys()))
        self.assertEqual(12, shapes[2]['area'])
        self.assertEqual(10, shapes[4]['area'])
        self.assertEqual(21, shapes[5]['area'])

        self.assertAlmostEqual(3.5, shapes[2]['pos_x'])
        self.assertAlmostEqual(5.5, shapes[4]['pos_x'])
        self.assertAlmostEqual(25, shapes[5]['pos_x'])

        self.assertEqual(3, shapes[2]['size_z'])
        self.assertEqual(2, shapes[4]['size_z'])
        self.assertEqual(1, shapes[5]['size_z'])

        self.assertEqual(False, 'solidity' in shapes[2])
        self.assertEqual(False, 'solidity' in shapes[4])
        self.assertEqual(False, 'solidity' in shapes[5])

        self.assertEqual(False, 'first_major_diff' in shapes[2])
        self.assertEqual(False, 'second_major_diff' in shapes[2])

    def test_extract_shape_features_resample_up(self):
        labels = self.generate_label_cases()
        labels[1, 26, 20:25] = 6  # object in bbox of 5

        # so far it is just smoke test
        shapes = feature_extractor.extract_shape_features(labels, 'complex', (1, 1, 1))
        self.assertEqual([2, 4, 5, 6], sorted(shapes.keys()))
        self.assertEqual(12, shapes[2]['area'])
        self.assertEqual(12, shapes[2]['volume_um'])
        self.assertEqual(10, shapes[4]['area'])
        self.assertEqual(7, shapes[4]['volume_um'])  # due to rounding
        self.assertEqual(21, shapes[5]['area'])
        self.assertEqual(21, shapes[5]['volume_um'])
        self.assertEqual(5, shapes[6]['area'])
        self.assertEqual(5, shapes[6]['volume_um'])

    def test_extract_shape_features_resample_down(self):
        labels = self.generate_label_cases()

        # so far it is just smoke test
        shapes = feature_extractor.extract_shape_features(labels, 'complex', (0.2, 0.3, 0.3))
        self.assertEqual([2, 4, 5], sorted(shapes.keys()))
        self.assertEqual(12, shapes[2]['area'])
        self.assertEqual(10, shapes[4]['area'])
        self.assertEqual(21, shapes[5]['area'])

    def test_get_channels_from_preprocessings(self):
        self.assertEqual([0], extractors.extract_channels(["0"]))
        self.assertEqual([0], extractors.extract_channels(["0-aw"]))
        self.assertEqual([0], extractors.extract_channels(["0", "0-aw"]))
        self.assertEqual([1, 3], extractors.extract_channels(["1-pl", "3-aa"]))

    def test_extract_complex_intensity_features(self):
        labels = self.generate_label_cases()
        intensity = np.random.random(labels.shape) / 5.0
        intensity[labels == 2] = 0.6
        intensity[1, 25, 20: 25] = 0.9

        marker_features = feature_extractor.extract_cells_features(intensity, labels, "complex")
        self.assertEqual([2, 4, 5], sorted(marker_features.keys()))
        self.assertTrue(marker_features[2]['moment_normalized_0_0_2'] > 0)
        self.assertTrue(marker_features[4]['moment_normalized_0_1_1'] > 0)
        self.assertTrue(marker_features[5]['moment_normalized_0_1_3'] < 0)

        self.assertAlmostEqual(0, marker_features[2]['mass_displace_in_diameters'], 2)
        self.assertAlmostEqual(0.20, marker_features[4]['mass_displace_in_diameters'], 2)
        self.assertAlmostEqual(0.548, marker_features[5]['mass_displace_in_diameters'], 2)

        self.assertAlmostEqual(0, marker_features[2]['mass_displace_in_majors'], 2)
        self.assertAlmostEqual(0.045, marker_features[4]['mass_displace_in_majors'], 2)
        self.assertAlmostEqual(0.145, marker_features[5]['mass_displace_in_majors'], 2)

    def test_extract_complex_texture_features(self):
        labels = self.generate_label_cases()
        intensity = np.random.random(labels.shape) / 5.0
        intensity[labels == 2] = 0.6
        intensity[1, 25, 20: 25] = 0.9

        marker_features = feature_extractor.extract_cells_features(intensity, labels, "complex+texture")
        self.assertEqual([2, 4, 5], sorted(marker_features.keys()))
        self.assertTrue(marker_features[2]['moment_normalized_0_0_2'] > 0)
        self.assertTrue(marker_features[4]['moment_normalized_0_1_1'] > 0)
        self.assertTrue(marker_features[5]['moment_normalized_0_1_3'] < 0)

        self.assertAlmostEqual(0, marker_features[2]['mass_displace_in_diameters'], 2)
        self.assertAlmostEqual(0.20, marker_features[4]['mass_displace_in_diameters'], 2)
        self.assertAlmostEqual(0.548, marker_features[5]['mass_displace_in_diameters'], 2)

        self.assertAlmostEqual(0, marker_features[2]['mass_displace_in_majors'], 2)
        self.assertAlmostEqual(0.045, marker_features[4]['mass_displace_in_majors'], 2)
        self.assertAlmostEqual(0.145, marker_features[5]['mass_displace_in_majors'], 2)

        self.assertNotEqual(0, marker_features[2]['haralick_mean_SumAverage'])
        self.assertEqual(0, marker_features[4]['haralick_mean_SumAverage'])  # it is weird shape
        self.assertNotEqual(0, marker_features[5]['haralick_mean_SumAverage'])

        self.assertAlmostEqual(0, marker_features[2]['mad_intensity'], 2)
        self.assertAlmostEqual(0.054, marker_features[4]['mad_intensity'], 2)
        self.assertAlmostEqual(0.054, marker_features[5]['mad_intensity'], 2)

    def test_all_contains_all_feature_types(self):
        labels = self.generate_label_cases()
        intensity = np.random.random(labels.shape) / 5.0
        intensity[labels == 2] = 0.6
        intensity[1, 25, 20: 25] = 0.9

        all_features = feature_extractor.extract_all_features(intensity, labels, features_type="all",
                                                              voxel_size=(2, 1, 1))

        for type in ['perc_per_10', 'complex', 'complex+texture', 'default', 'default+texture']:
            type_features = feature_extractor.extract_all_features(intensity, labels, features_type=type,
                                                                   voxel_size=(2, 1, 1))
            only_features_columns = feature_extractor.get_feature_columns(type_features[2].keys(), type)
            only_features_columns_from_all = feature_extractor.get_feature_columns(all_features[2].keys(), type)

            self.assertEqual(only_features_columns, only_features_columns_from_all)

            only_features_data = self.filter_features_to_keys(type_features, only_features_columns)
            only_features_data_all = self.filter_features_to_keys(all_features, only_features_columns_from_all)

            self.assertEqual(only_features_data, only_features_data_all)

    def test_same_features_all_or_separately(self):
        labels = self.generate_label_cases()
        intensity = np.random.random(list(labels.shape) + [2]) / 5.0
        intensity[labels == 2] = 0.6
        intensity[1, 25, 20: 25] = 0.9
        preprocesses = "0,0-clahe,0-edges,0-memb,1,1-clahe,1-edges,1-memb"
        prelist = extractors.parse_channels_preprocessing(preprocesses)

        original_intensity = intensity.copy()
        original_labels = labels.copy()

        all_features_for_all = feature_extractor.extract_all_features(intensity, labels,
                                                                      channels_with_preprocessing_list=prelist,
                                                                      features_type="all",
                                                                      voxel_size=None)

        for preprocess in prelist:
            all_features_for_preprocess = feature_extractor.extract_all_features(intensity, labels,
                                                                                 channels_with_preprocessing_list=[
                                                                                     preprocess],
                                                                                 features_type="all",
                                                                                 voxel_size=None)

            only_features_columns = feature_extractor.get_feature_columns(all_features_for_preprocess[2].keys(),
                                                                          "all", [preprocess])
            only_features_columns_from_all = feature_extractor.get_feature_columns(all_features_for_all[2].keys(),
                                                                                   "all", [preprocess])
            self.assertEqual(only_features_columns, only_features_columns_from_all)

            only_features_data = self.filter_features_to_keys(all_features_for_preprocess, only_features_columns)
            only_features_data_all = self.filter_features_to_keys(all_features_for_all, only_features_columns_from_all)

            self.assertEqual(only_features_data, only_features_data_all)

        nptest.assert_array_equal(original_intensity, intensity)
        nptest.assert_array_equal(original_labels, labels)

    def tearDown(self):
        pass
