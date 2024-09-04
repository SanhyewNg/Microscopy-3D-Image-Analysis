import pytest
import unittest
import unittest.mock as mock

import bioformats

from clb.dataprep.lif.utils import (ChannelMeta,
                                     ImageMeta,
                                     MarkerMismatchWarning,
                                     PixelsMeta,
                                     PlaneMeta,
                                     split_name,
                                     UnrecognizedParametersError)


@pytest.mark.io
class TestLifUtils(unittest.TestCase):
    def test_if_split_name_raises_exception_when_given_wrong_order(self):
        with self.assertRaises(UnrecognizedParametersError):
            split_name('sample_slice')

    def test_if_split_name_gives_right_output(self):
        name = 'sample_size_speed_region_slice'

        output = split_name(name)
        right_output = {'sample': 'sample', 'size': 'size',
                        'speed': 'speed',
                        'region': 'region', 'slice': 'slice'}

        self.assertDictEqual(output, right_output)


test_ome = bioformats.OMEXML('''
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06
http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
<Image Name="Image0">
<Pixels>
<Channel/>
<Channel Color="65535" ExcitationWavelength="52.213"/>
<Channel Color="-65281"/>
<Channel Color="16711935"/>
<Channel Color="-16776961"/>
<Channel Color="1"/>
</Pixels>
</Image>
<Image Name="Image1">
<Pixels PhysicalSizeX="123.45" PhysicalSizeY="234.56" PhysicalSizeZ="345.67"
 PhysicalSizeXUnit="cm" PhysicalSizeYUnit="mm" PhysicalSizeZUnit="m"
 Type="uint8">
<Channel Color="65535" ExcitationWavelength="123.45"/>
<Channel Color="-65281"/>
<Channel Color="16711935"/>
<Channel Color="-16776961"/>
<Plane TheC="0" TheZ="0"/>
</Pixels>
</Image>
</OME>
''')


@pytest.mark.io
class TestChannelMeta(unittest.TestCase):
    def test__get_color_output_when_no_color(self):
        ome_meta = test_ome.image(0).Pixels.Channel(0)

        output = ChannelMeta._get_color(ome_meta)

        self.assertIsNone(output)

    def test__get_color_output(self):
        ome_meta = test_ome.image(0).Pixels.Channel(1)

        output = ChannelMeta._get_color(ome_meta)
        right_output = (0, 0, 255)

        self.assertEqual(output, right_output)

    def test__get_exc_wavelen_output_when_no_color(self):
        ome_meta = test_ome.image(0).Pixels.Channel(0)

        output = ChannelMeta._get_exc_wavelen(ome_meta)

        self.assertIsNone(output)

    def test__get_exc_wavelen_output(self):
        ome_meta = test_ome.image(0).Pixels.Channel(1)

        output = ChannelMeta._get_exc_wavelen(ome_meta)
        right_output = 52.213

        self.assertEqual(output, right_output)

    def test_if_warning_is_raised_when_color_and_index_marker_dont_match(self):
        panck_index = 1
        dapi_color = 65535

        with self.assertWarns(MarkerMismatchWarning):
            ChannelMeta._get_marker(panck_index, dapi_color)

    def test__get_marker_all_outputs(self):
        dapi_index, dapi_color = 0, 65535
        panck_index, panck_color = 1, -65281
        ki_index, ki_color = 2, 16711935
        cd_index, cd_color = 3, -16776961

        self.assertEqual(ChannelMeta._get_marker(dapi_index, dapi_color),
                         'dapi')
        self.assertEqual(ChannelMeta._get_marker(panck_index, panck_color),
                         'pan-cytokeratin')
        self.assertEqual(ChannelMeta._get_marker(ki_index, ki_color),
                         'ki67')
        self.assertEqual(ChannelMeta._get_marker(cd_index, cd_color),
                         'cd3')

    def test_get_meta_output(self):
        ome_meta = test_ome.image(1).Pixels.Channel(0)
        channel_meta = ChannelMeta(ome_meta, 0)

        output = channel_meta.get_meta()
        right_output = {
            'channel_index': 0,
            'color': (0, 0, 255),
            'marker': 'dapi',
            'exc_wavelen': 123.45
        }

        self.assertEqual(output, right_output)


@pytest.mark.io
class TestPlaneMeta(unittest.TestCase):
    def test__get_channel_output(self):
        ome_meta = test_ome.image(1).Pixels.Plane(0)

        output = PlaneMeta._get_channel(ome_meta)
        right_output = 0

        self.assertEqual(output, right_output)

    def test__get_z(self):
        ome_meta = test_ome.image(1).Pixels.Plane(0)

        output = PlaneMeta._get_z(ome_meta)
        right_output = 0

        self.assertEqual(output, right_output)

    def test_get_meta(self):
        ome_meta = test_ome.image(1).Pixels.Plane(0)
        plane_meta = PlaneMeta(ome_meta, 0)

        output = plane_meta.get_meta()
        right_output = {
            'plane_index': 0,
            'z': 0
        }

        self.assertEqual(output, right_output)


@pytest.mark.io
class TestPixelsMeta(unittest.TestCase):
    def test__get_size_x(self):
        ome_meta = test_ome.image(1).Pixels

        output = PixelsMeta._get_size_x(ome_meta)
        right_output = 123.45

        self.assertEqual(output, right_output)

    def test__get_size_y(self):
        ome_meta = test_ome.image(1).Pixels

        output = PixelsMeta._get_size_y(ome_meta)
        right_output = 234.56

        self.assertEqual(output, right_output)

    def test__get_size_z(self):
        ome_meta = test_ome.image(1).Pixels

        output = PixelsMeta._get_size_z(ome_meta)
        right_output = 345.67

        self.assertEqual(output, right_output)

    def test__get_size_x_unit(self):
        ome_meta = test_ome.image(1).Pixels

        output = PixelsMeta._get_size_x_unit(ome_meta)
        right_output = 'cm'

        self.assertEqual(output, right_output)

    def test__get_size_y_unit(self):
        ome_meta = test_ome.image(1).Pixels

        output = PixelsMeta._get_size_y_unit(ome_meta)
        right_output = 'mm'

        self.assertEqual(output, right_output)

    def test__get_size_z_unit(self):
        ome_meta = test_ome.image(1).Pixels

        output = PixelsMeta._get_size_z_unit(ome_meta)
        right_output = 'm'

        self.assertEqual(output, right_output)

    def test__get_type(self):
        ome_meta = test_ome.image(1).Pixels

        output = PixelsMeta._get_type(ome_meta)
        right_output = 'uint8'

        self.assertEqual(output, right_output)

    def test___getitem___if_it_calls_get_meta_methods(self):
        with mock.patch('clb.dataprep.lif.utils.PixelsMeta.get_meta')\
             as mock_pix,\
             mock.patch('clb.dataprep.lif.utils.ChannelMeta.get_meta')\
             as mock_ch,\
             mock.patch('clb.dataprep.lif.utils.PlaneMeta.get_meta')\
             as mock_plane:
            ome_meta = test_ome.image(1).Pixels
            pixels_meta = PixelsMeta(ome_meta)

            _ = pixels_meta[0]

            mock_pix.assert_any_call()
            mock_ch.assert_any_call()
            mock_plane.assert_any_call()

    def test___getitem___if_output_is_sum_of_get_meta_outputs(self):
        with mock.patch('clb.dataprep.lif.utils.PixelsMeta.get_meta',
                        new=mock.Mock(return_value=dict(a=1))),\
             mock.patch('clb.dataprep.lif.utils.ChannelMeta.get_meta',
                        new=mock.Mock(return_value=dict(b=2))),\
             mock.patch('clb.dataprep.lif.utils.PlaneMeta.get_meta',
                        new=mock.Mock(return_value=dict(c=3))):
            ome_meta = test_ome.image(1).Pixels
            pixels_meta = PixelsMeta(ome_meta)

            output = pixels_meta[0]
            right_output = dict(a=1, b=2, c=3)

            self.assertEqual(output, right_output)

    def test_get_meta_output(self):
        ome_meta = test_ome.image(1).Pixels
        pixels_meta = PixelsMeta(ome_meta)

        output = pixels_meta.get_meta()
        right_output = {
            'size_x': 123.45,
            'size_y': 234.56,
            'size_z': 345.67,
            'size_x_unit': 'cm',
            'size_y_unit': 'mm',
            'size_z_unit': 'm',
            'type': 'uint8'
        }

        self.assertEqual(output, right_output)


@pytest.mark.io
class TestImageMeta(unittest.TestCase):
    def test___getitem___if_it_calls_get_meta_and_getitem(self):
        with mock.patch('clb.dataprep.lif.utils.ImageMeta.get_meta')\
             as mock_im,\
             mock.patch('clb.dataprep.lif.utils.PixelsMeta.__getitem__')\
             as mock_pix:
            ome_meta = test_ome.image(1)
            image_meta = ImageMeta(ome_meta, 1)

            _ = image_meta[0]

            mock_im.assert_any_call()
            mock_pix.assert_called_once_with(0)

    def test___getitem___if_output_is_sum_of_get_meta_and_getitem(self):
        with mock.patch('clb.dataprep.lif.utils.ImageMeta.get_meta',
                        new=mock.Mock(return_value=dict(a=1))),\
             mock.patch('clb.dataprep.lif.utils.PixelsMeta.__getitem__',
                        new=mock.Mock(return_value=dict(b=2))):
            ome_meta = test_ome.image(1)
            image_meta = ImageMeta(ome_meta, 1)

            output = image_meta[0]
            right_output = dict(a=1, b=2)

            self.assertEqual(output, right_output)

    def test_get_meta(self):
        ome_meta = test_ome.image(1)
        image_meta = ImageMeta(ome_meta, 1)

        output = image_meta.get_meta()
        right_output = {'series': 1}

        self.assertEqual(output, right_output)


if __name__ == '__main__':
    unittest.main()
