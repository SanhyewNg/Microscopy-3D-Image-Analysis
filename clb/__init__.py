import imageio

from clb.dataprep.imageioplug.lifformat import LifFormat
from clb.dataprep.imageioplug.tifformat import TiffFormat


tiff_format = TiffFormat(
    name='TiffFormat',
    description='Plugin for reading .tif files using bioformats.',
    extensions='.tif .tiff .btf',
    modes='v',
)


lif_format = LifFormat(
    name='LifFormat',
    description='Plugin for reading .lif files using bioformats.',
    extensions='.lif',
    modes='v',
)

imageio.formats.add_format(tiff_format)
imageio.formats.add_format(lif_format)
imageio.formats.sort('TiffFormat', 'LifFormat')