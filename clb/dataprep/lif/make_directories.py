"""Module defines functions for creating directory tree with images.

Attributes:
    default_layout (tuple): Layout of directories. Each element of tuple is an
                            OrderedDict mapping parameter name to it's label in
                            name of file or directory. {} in each label will be
                            replaced with parameter value. See `create_names`
                            function docstring for more info.

Future task: Add function for grouping metadata (describing data and resizing).
"""
from collections import OrderedDict
import os
import warnings

import bioformats
import cv2
import fire
import javabridge

from clb.dataprep.lif.lif_readers import DenoisingLifReader, MetadataError


default_layout = (
    OrderedDict([('sample', 'sample-{}'),
                 ('slice', 'z{}'),
                 ('region', 'region-{}'),
                 ('marker', 'marker-{}'),
                 ]),
    OrderedDict([('size', 'size-{}px'),
                 ('speed', 'speed-{}Hz'),
                 ('averaging_steps', 'averaging_steps-{}')
                 ])
)


def create_names(layout, params, separator):
    """Create names of directories from given layout and parameters.

    Let's say `layout` is like this:
    (
    OrderedDict([('sample', 'sample-{}'),
                 ('slice', 'z{}'),
                 ('region', 'region-{}'),
                 ('marker', 'marker-{}')
                 ]),
    OrderedDict([('size', 'size-{}px'),
                 ('speed', 'speed-{}Hz'),
                 ('averaging_steps', 'averaging_steps-{}')
                 ]),
    )
    `params` are {'sample': 'Tonsil2',
                  'slice': 1,
                  'region: 'FOV2',
                  'marker': 'dapi',
                  'size': 400,
                  'speed': 600,
                  'averaging_steps': 4
                  }
    and `separator` is '_'. Then function should yield
    ('sample-Tonsil2_z1_region-FOV2_marker-dapi',
    'size-400px_speed-600Hz_averaging_steps-4'). First element is directory
    name, last is file name.

    Args:
        layout (tuple): Layout of directories.
        params (dict): Mapping parameter names to their values.
        separator (str): Separator used in names of files and directories.

    Yields:
        str: Names of directories.
    """
    for piece in layout:
        name_pieces = (label.format(params[param_name])
                       for param_name, label in piece.items())
        name = separator.join(name_pieces)
        yield name


def make_directories_from_file(path, save_dir='lif_images_trial1',
                               layout=default_layout, params=None, ext='.png',
                               separator='_', resize=True):
    """Create directory tree described in `layout`, read images from file

    path (str): Path to .lif file.
    save_dir (str): Path for saving images.
    layout (tuple): Layout of directories.
    params (dict): Parameters of images to read. If None all images are
                   read.
    ext (str): Extension of file.
    separator (str): Separator of information in names.
    resize (bool): Should image be upscaled to 2048x2048.

    Future task: Add tests.
    Future task: Control behavior when file or directory exists.
    Future task: Resizing in groups, instead of hardcoded 2048.
    """
    # If there are no params we should read all images, so no criteria.
    if params is None:
        params = {}

    reader = DenoisingLifReader()
    data = reader.read_data_given_meta(path, params)

    for meta, image in data:
        # Creating names of directories for image.
        names = list(create_names(layout, meta, separator))

        # Creating directories path.
        save_path = os.path.join(save_dir, *names[:-1])

        # Creating directories for image.
        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass

        # Writing image.
        if resize:
            image = cv2.resize(image, (2048, 2048))

        bioformats.write_image(os.path.join(save_path, names[-1] + ext),
                               image, meta['pixel_type'])


def make_directories_from_dir(load_dir, save_dir, layout=default_layout,
                              params=None, ext='.png', separator='_',
                              resize=True):
    """Create directory tree described in `layout`, read images from directory.

    load_dir (str): Path to directory with .lif files.
    save_dir (str): Path for saving images.
    layout (tuple): Layout of directories.
    params (dict): Parameters of images to read. If None all images are
                   read.
    ext (str): Extension of file.
    separator (str): Separator of information in names.
    resize (bool): Should image be upscaled to 2048x2048.
    """
    dirs = os.listdir(load_dir)
    for path in dirs:
        try:
            make_directories_from_file(os.path.join(load_dir, path), save_dir,
                                       layout, params, ext, separator, resize)
        except MetadataError:
            warnings.warn("Couldn't read file {}".format(path))


if __name__ == '__main__':
    javabridge.start_vm(class_path=bioformats.JARS)

    fire.Fire(make_directories_from_dir)

    javabridge.kill_vm()
