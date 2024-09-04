import os
import pytest
import shutil
import tempfile
import unittest.mock as mock

import clb.dataprep.tif.tif_readers as tif_readers
import clb.stats.all_stats as all_stats
import clb.stats.intensity_stats as int_stats
import clb.stats.morphology_stats as morph_stats
import clb.stats.volume_stats as vol_stats
import tests.utils as tutils


@pytest.mark.statistics
class TestStats(tutils.TestCase):
    def setUp(self):
        self.images_path = 'tests/test_images/lif'
        self.series_0 = 'tests/test_images/lif/series_0.tif'
        self.series_1 = 'tests/test_images/lif/series_1.tif'
        self.series_0_labels = 'tests/test_images/lif/series_0_labels.tif'
        self.series_1_labels = 'tests/test_images/lif/series_1_labels.tif'
        self.series_0_classes_1 = 'tests/test_images/lif/series_0_classes_1.tif'
        self.series_0_classes_2 = 'tests/test_images/lif/series_0_classes_2.tif'
        self.series_1_classes_1 = 'tests/test_images/lif/series_0_classes_1.tif'
        self.series_1_classes_1 = 'tests/test_images/lif/series_0_classes_2.tif'
        self.stats_path = 'tests/test_statistics'

        self.results_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results_dir)

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
                reader.get_metadata = mock.Mock(return_value=reader.metadata)

            return reader

        patch_get_volume_reader = mock.patch(
            target='clb.stats.volume_stats.readers._get_reader',
            new=mock_get_reader)
        patch_get_volume_reader.start()
        self.addCleanup(patch_get_volume_reader.stop)

    def test_volume_stats(self):
        output_path = os.path.join(self.results_dir, '{name}_volume_stats.csv')
        vol_stats.main(input='input.lif',
                       labels=os.path.join(self.images_path, '{name}_labels.tif'),
                       output=output_path,
                       series=0,
                       ki67=os.path.join(self.images_path, '{name}_classes_1.tif'),
                       epith=os.path.join(self.images_path, '{name}_classes_2.tif'))

        out_path = os.path.join(self.results_dir, 'series_0_volume_stats.csv')
        path_of_expected = os.path.join(self.stats_path, 'series_0_volume_stats.csv')
        self.assert_csv_equal_to_csv(csv_path1=out_path,
                                     csv_path2=path_of_expected)

    def test_morphology_stats(self):
        output_path = os.path.join(self.results_dir, '{name}_morphology_stats.csv')
        morph_stats.main(input='input.lif',
                         labels=os.path.join(self.images_path, '{name}_labels.tif'),
                         output=output_path,
                         series=0,
                         ki67=os.path.join(self.images_path, '{name}_classes_1.tif'),
                         epith=os.path.join(self.images_path, '{name}_classes_2.tif'))

        out_path = os.path.join(self.results_dir, 'series_0_morphology_stats.csv')
        path_of_expected = os.path.join(self.stats_path, 'series_0_morphology_stats.csv')
        self.assert_csv_equal_to_csv(csv_path1=out_path,
                                     csv_path2=path_of_expected)

    def test_intensity_stats(self):
        output_path = os.path.join(self.results_dir, '{name}_intensity_stats.csv')
        int_stats.main(input='input.lif',
                       labels=os.path.join(self.images_path, '{name}_labels.tif'),
                       output=output_path,
                       channels=[0, 1, 2],
                       series=1,
                       channel_names=['dapi', 'ki67', 'panck'],
                       ki67=os.path.join(self.images_path, '{name}_classes_1.tif'),
                       epith=os.path.join(self.images_path, '{name}_classes_2.tif'))

        out_path = os.path.join(self.results_dir, 'series_1_intensity_stats.csv')
        path_of_expected = os.path.join(self.stats_path, 'series_1_intensity_stats.csv')
        self.assert_csv_equal_to_csv(csv_path1=out_path,
                                     csv_path2=path_of_expected)

    def test_all_stats(self):
        output_path = os.path.join(self.results_dir, '{name}_')
        all_stats.main(input='input.lif',
                       labels=os.path.join(self.images_path, '{name}_labels.tif'),
                       output=output_path,
                       channels=[0, 1, 2],
                       channel_names=['dapi', 'ki67', 'panck'],
                       ki67=os.path.join(self.images_path, '{name}_classes_1.tif'),
                       epith=os.path.join(self.images_path, '{name}_classes_2.tif'),
                       append_aggregations=True,
                       merge=True,
                       series=0)

        out_path = os.path.join(self.results_dir, 'series_0_volume_stats.csv')
        path_of_expected = os.path.join(self.stats_path,
                                        'series_0_volume_stats_with_aggregations.csv')
        self.assert_csv_equal_to_csv(csv_path1=out_path,
                                     csv_path2=path_of_expected)

        out_path = os.path.join(self.results_dir, 'series_0_morphology_stats.csv')
        path_of_expected = os.path.join(self.stats_path, 'series_0_morphology_stats.csv')
        self.assert_csv_equal_to_csv(csv_path1=out_path,
                                     csv_path2=path_of_expected)

        out_path = os.path.join(self.results_dir, 'series_0_nuclei_stats.csv')
        path_of_expected = os.path.join(self.stats_path, 'series_0_nuclei_stats.csv')
        self.assert_csv_equal_to_csv(csv_path1=out_path,
                                     csv_path2=path_of_expected)

        out_path = os.path.join(self.results_dir, 'series_0_intensity_stats.csv')
        path_of_expected = os.path.join(self.stats_path, 'series_0_intensity_stats.csv')
        self.assert_csv_equal_to_csv(csv_path1=out_path,
                                     csv_path2=path_of_expected)
