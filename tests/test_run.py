import argparse
import functools as ft
import os
import pytest
import shutil
import tempfile
import unittest
import unittest.mock as mock

import numpy as np
import skimage

import clb.dataprep.tif.tif_readers as tif_readers
import clb.run as run
import clb.segment.segment_cells as segment
import tests.utils as tutils


@pytest.mark.integration
class TestTif(tutils.TestCase):
    def setUp(self):
        self.dir = 'tests/test_images/tif'
        self.output_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.output_dir)

        self.one_ch = os.path.join(self.dir, 'one_channel.tif')
        self.rgb = os.path.join(self.dir, 'rgb.tif')
        self.one_ch_labels = os.path.join(self.dir, 'one_channel_labels.tif')
        self.rgb_labels = os.path.join(self.dir, 'rgb_labels.tif')

        def threshold(images, model, model_args, limit=0.5):
            float_images = map(skimage.img_as_float64, images)
            thresholded = [np.where(image > limit, image, 0)
                           for image in float_images]
            ubyte_images = map(skimage.img_as_ubyte, thresholded)
            squeezed = map(np.squeeze, ubyte_images)

            return squeezed

        patch_segment = mock.patch('clb.predict.predict3d.predict',
                                   new=threshold)
        patch_segment.start()
        self.addCleanup(patch_segment.stop)

        cc_method = ft.partial(segment.label_cells_cc, threshold=0.5,
                               opening=1, dilation=2)
        mock_make_watershed_method = mock.Mock(return_value=cc_method)
        patch_make_watershed_method = mock.patch(
            target='clb.run.make_watershed_method',
            new=mock_make_watershed_method)
        patch_make_watershed_method.start()
        self.addCleanup(patch_make_watershed_method.stop)

        self.parser = run.get_parser()
        self.model_path = 'models/model_7_eager_torvalds.h5'
        self.args = argparse.Namespace(model=self.model_path,
                                       use_channel=0,
                                       start=0,
                                       stop=None,
                                       pixel_size=(0.5, 0.5),
                                       desired_pixel_size=(0.5, 0.5),
                                       resize_tolerance=0.05,
                                       no_pixel_resize=False,
                                       series=None)

    def test_one_channel_tif(self):
        output_path = os.path.join(self.output_dir, 'output.tif')
        self.args.input = self.one_ch
        self.args.outputs = [output_path]

        run.main(self.args)

        self.assert_tif_equal_to_tif(output_path, self.one_ch_labels)

    def test_rgb(self):
        output_path = os.path.join(self.output_dir, 'output.tif')
        self.args.input = self.rgb
        self.args.outputs = [output_path]

        run.main(self.args)

        self.assert_tif_equal_to_tif(output_path, self.rgb_labels)

    def test_rgb_with_ims_output(self):
        output_path = os.path.join(self.output_dir, 'output.ims')
        self.args.input = self.rgb
        self.args.outputs = [output_path]

        run.main(self.args)

        self.assert_ims_equal_to_tifs(output_path,
                                      (self.rgb, 0),
                                      self.rgb_labels)

    def test_rgb_with_tif_and_ims_output(self):
        output_path = os.path.join(self.output_dir, 'output')
        os.makedirs(output_path)
        tif_output_path = os.path.join(output_path, 'labels.tif')
        ims_output_path = os.path.join(output_path, 'labels.ims')
        self.args.input = self.rgb
        self.args.outputs = [tif_output_path, ims_output_path]

        run.main(self.args)

        self.assert_tif_equal_to_tif(tif_output_path, self.rgb_labels)
        self.assert_ims_equal_to_tifs(ims_output_path,
                                      (self.rgb, 0),
                                      self.rgb_labels)


@pytest.mark.integration
class TestLif(tutils.TestCase):
    def setUp(self):
        self.series_0 = 'tests/test_images/lif/series_0.tif'
        self.series_1 = 'tests/test_images/lif/series_1.tif'
        self.series_0_labels = 'tests/test_images/lif/series_0_labels.tif'
        self.series_1_labels = 'tests/test_images/lif/series_1_labels.tif'

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
                    reader._metadata = {'Name': 'series_0'}
                else:
                    reader = tif_readers.TifReader(self.series_1)
                    reader._metadata = {'Name': 'series_1'}

                reader.dimensions['s'] = 2
                reader._metadata.update(metadata)

            return reader

        patch_get_volume_reader = mock.patch(
            target='clb.classify.classify.readers._get_reader',
            new=mock_get_reader)
        patch_get_volume_reader.start()
        self.addCleanup(patch_get_volume_reader.stop)

        self.output_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.output_dir)

        self.parser = run.get_parser()
        self.model_path = 'models/model_7_eager_torvalds.h5'

        def threshold(images, model, model_args, limit=0.5):
            float_images = map(skimage.img_as_float64, images)
            thresholded = (np.where(image > limit, image, 0)
                           for image in float_images)
            ubyte_images = map(skimage.img_as_ubyte, thresholded)
            squeezed = map(np.squeeze, ubyte_images)

            return squeezed

        patch_segment = mock.patch('clb.predict.predict3d.predict',
                                   new=threshold)
        patch_segment.start()
        self.addCleanup(patch_segment.stop)

        cc_method = ft.partial(segment.label_cells_cc, threshold=0.5,
                               opening=1, dilation=2)
        mock_make_watershed_method = mock.Mock(return_value=cc_method)
        patch_make_watershed_method = mock.patch(
            target='clb.run.make_watershed_method',
            new=mock_make_watershed_method)
        patch_make_watershed_method.start()
        self.addCleanup(patch_make_watershed_method.stop)

        args = argparse.Namespace(input='input.lif',
                                  model=self.model_path,
                                  use_channel=0,
                                  start=0,
                                  stop=None,
                                  pixel_size=(0.5, 0.5),
                                  desired_pixel_size=(0.5, 0.5),
                                  resize_tolerance=0.05,
                                  no_pixel_resize=False)
        self.args = args

    def test_one_series_with_tif_output(self):
        output_path = os.path.join(self.output_dir, 'series_0.tif')
        self.args.outputs = [output_path]
        self.args.series = 0

        run.main(self.args)

        self.assert_tif_equal_to_tif(output_path, self.series_0_labels)

    def test_if_resize_to_pixel_size_is_called(self):
        def resize(image, pixel_size, desired_pixel_size):
            return image
        mock_resize = mock.Mock(side_effect=resize)
        with mock.patch(
                'clb.dataprep.utils.resize_to_pixel_size',
                new=mock_resize) as mock_resize:
            output_path = os.path.join(self.output_dir, 'series_0.ims')
            self.args.outputs = [output_path]
            self.args.series = [0]
            self.args.desired_pixel_size = (0.8, 0.8)
            run.main(self.args)
            call_list = [mock.call(mock.ANY, (0.5, 0.5), (0.8, 0.8))
                         for _ in range(10)]
            mock_resize.assert_has_calls(call_list)
        # patch_get_volume_reader.stop()

    def test_one_series_with_ims_output(self):
        output_path = os.path.join(self.output_dir, 'series_1.ims')
        self.args.outputs = [output_path]
        self.args.series = [1]

        run.main(self.args)

        self.assert_ims_equal_to_tifs(output_path,
                                      (self.series_1, 0),
                                      self.series_1_labels)

    def test_one_series_with_ims_output_with_start_stop(self):
        output_path = os.path.join(self.output_dir, 'series_1.ims')
        self.args.outputs = [output_path]
        self.args.series = [1]
        self.args.start = 0
        self.args.stop = 4

        run.main(self.args)

        self.assert_ims_equal_to_tifs(output_path,
                                      (self.series_1, 0),
                                      self.series_1_labels,
                                      slices=slice(self.args.start,
                                                   self.args.stop))
