import pytest
import unittest

import SimpleITK as sitk
import numpy as np
import numpy.testing as nptest
import pandas as pd

import clb.stats.morphology_stats as ms


@pytest.mark.statistics
class TestCalcCellVolume(unittest.TestCase):
    def test_with_ones_sizes(self):
        cell_mask = np.array([
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
        ])
        sizes = (1, 1, 1)

        image = sitk.GetImageFromArray(cell_mask)
        image.SetSpacing(spacing=sizes)
        feature_extractor = ms.ExtendedRadiomicsShape(image, image)
        volume = ms.calc_cell_volume(feature_extractor)
        expected_volume = 5

        self.assertEqual(volume, expected_volume)

    def test_with_different_sizes(self):
        cell_mask = np.array([
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
        ])
        sizes = (0.5, 0.5, 0.2)

        image = sitk.GetImageFromArray(cell_mask)
        image.SetSpacing(spacing=sizes)
        feature_extractor = ms.ExtendedRadiomicsShape(image, image)
        volume = ms.calc_cell_volume(feature_extractor)
        expected_volume = 0.4

        self.assertEqual(volume, expected_volume)


@pytest.mark.statistics
class TestCalcEquivalentDiameter(unittest.TestCase):
    def test_output(self):
        volume = 8

        diameter = ms.calc_equivalent_diameter(volume)
        expected_diameter = 1.24070098179

        nptest.assert_almost_equal(diameter, expected_diameter)


@pytest.mark.statistics
class TestCalcProlateness(unittest.TestCase):
    def test_output(self):
        eigenvalues = np.array([1, 2, 3])

        prolateness = ms.calc_prolateness(eigenvalues)
        expected_prolateness = (np.sqrt(3.) - np.sqrt(1.)) / np.sqrt(3.)

        self.assertEqual(prolateness, expected_prolateness)

    def test_if_exception_is_raised_with_negative_eigenvalues(self):
        eigenvalues = np.array([3, 2, -1])

        with self.assertRaises(ms.NegativeEigenvaluesError):
            ms.calc_prolateness(eigenvalues)


@pytest.mark.statistics
class TestCalcOblateness(unittest.TestCase):
    def test_output(self):
        eigenvalues = np.array([1, 2, 3])

        oblateness = ms.calc_oblateness(eigenvalues)
        expected_oblateness = 2 * ((np.sqrt(2.) - np.sqrt(1.))
                                   / (np.sqrt(3.) - np.sqrt(1.))) - 1

        self.assertEqual(oblateness, expected_oblateness)

    def test_if_exception_is_raised_with_negative_eigenvalues(self):
        eigenvalues = np.array([3, 2, -1])

        with self.assertRaises(ms.NegativeEigenvaluesError):
            ms.calc_oblateness(eigenvalues)


@pytest.mark.statistics
class TestAddAggregation(unittest.TestCase):
    def test_aggregating_one_feature_mean(self):
        morphology_stats_dict = {
            'classes': ['no_class', 'a, b', 'no_class', 'b', 'no_class'],
            'sphericity': [1, 2, 2, 3, 5]
        }
        morphology_stats = pd.DataFrame.from_dict(morphology_stats_dict)

        volume_stats_with_aggregation = ms.calc_aggregation(morphology_stats,
                                                            aggregation_type='mean',
                                                            features=('sphericity',))
        expected_dict = {
            'classes': ['no_class', 'b', 'a, b', 'all_cells'],
            'mean_sphericity': [8/3, 3, 2, 13/5]
        }
        expected = pd.DataFrame.from_dict(expected_dict).set_index('classes')

        pd.testing.assert_frame_equal(volume_stats_with_aggregation.sort_index(),
                                      expected.sort_index())

    def test_aggregation_two_features_std(self):
        morphology_stats_dict = {
            'classes': ['no_class', 'a, b', 'no_class', 'b', 'b', 'no_class'],
            'sphericity': [1, 2, 2, 3, 4, 5],
            'prolateness': [0, 2, 1, 3, 4, 5]
        }
        morphology_stats = pd.DataFrame.from_dict(morphology_stats_dict)

        volume_stats_with_aggregation = ms.calc_aggregation(morphology_stats,
                                                            aggregation_type='std',
                                                            features=('sphericity',
                                                                      'prolateness'))
        expected_dict = {
            'classes': ['no_class', 'b', 'a, b', 'all_cells'],
            'std_sphericity': [np.std([1, 2, 5]),
                               np.std([3, 4]),
                               np.std([2]),
                               np.std([1, 2, 2, 3, 4, 5])],
            'std_prolateness': [np.std([0, 1, 5]),
                                np.std([3, 4]),
                                np.std([2]),
                                np.std([0, 2, 1, 3, 4, 5])]
        }
        expected = pd.DataFrame.from_dict(expected_dict).set_index('classes')
        expected.sort_index(inplace=True)
        expected = expected.reindex(columns=sorted(expected.columns))
        volume_stats_with_aggregation.sort_index(inplace=True)
        volume_stats_with_aggregation = volume_stats_with_aggregation.reindex(
            columns=sorted(volume_stats_with_aggregation.columns)
        )

        pd.testing.assert_frame_equal(volume_stats_with_aggregation.sort_index(),
                                      expected.sort_index())

    def test_aggregation_two_features_median(self):
        morphology_stats_dict = {
            'classes': ['no_class', 'a, b', 'no_class', 'b', 'b', 'no_class'],
            'sphericity': [1, 2, 2, 3, 4, 5],
            'prolateness': [0, 2, 1, 3, 4, 5]
        }
        morphology_stats = pd.DataFrame.from_dict(morphology_stats_dict)

        volume_stats_with_aggregation = ms.calc_aggregation(morphology_stats,
                                                            aggregation_type='median',
                                                            features=('sphericity',
                                                                      'prolateness'))
        expected_dict = {
            'classes': ['no_class', 'b', 'a, b', 'all_cells'],
            'median_sphericity': [np.median([1, 2, 5]),
                                  np.median([3, 4]),
                                  np.median([2]),
                                  np.median([1, 2, 2, 3, 4, 5])],
            'median_prolateness': [np.median([0, 1, 5]),
                                   np.median([3, 4]),
                                   np.median([2]),
                                   np.median([0, 2, 1, 3, 4, 5])]
        }
        expected = pd.DataFrame.from_dict(expected_dict).set_index('classes')
        expected.sort_index(inplace=True)
        expected = expected.reindex(columns=sorted(expected.columns))
        volume_stats_with_aggregation.sort_index(inplace=True)
        volume_stats_with_aggregation = volume_stats_with_aggregation.reindex(
            columns=sorted(volume_stats_with_aggregation.columns)
        )

        pd.testing.assert_frame_equal(volume_stats_with_aggregation.sort_index(),
                                      expected.sort_index())


@pytest.mark.statistics
class TestAppendAggregations(unittest.TestCase):
    def test_one_column(self):
        volume_stats_dict = {
            'classes': ['no_class', 'a, b', 'a', 'b', 'all_cells'],
            'volume': [10, 5, 6, 7, 15]
        }
        volume_stats = pd.DataFrame.from_dict(volume_stats_dict)
        volume_stats.set_index('classes', inplace=True)
        features_aggregations_dict = {
            'classes': ['all_cells', 'a', 'b', 'a, b', 'no_class'],
            'mean_sphericity': [4, 2, 3, 1, 2]
        }
        features_aggregations = pd.DataFrame.from_dict(features_aggregations_dict)
        features_aggregations.set_index('classes', inplace=True)

        stats_with_aggregations = ms.add_aggregations(volume_stats,
                                                      features_aggregations)
        stats_with_aggregations = stats_with_aggregations.reindex(
            columns=sorted(stats_with_aggregations.columns))
        expected_dict = {
            'classes': ['no_class', 'a, b', 'a', 'b', 'all_cells'],
            'volume': [10, 5, 6, 7, 15],
            'mean_sphericity': [2, 1, 2, 3, 4]
        }
        expected = pd.DataFrame.from_dict(expected_dict)
        expected.set_index('classes', inplace=True)
        expected = expected.reindex(columns=sorted(expected.columns))

        pd.testing.assert_frame_equal(stats_with_aggregations.sort_index(),
                                      expected.sort_index())
