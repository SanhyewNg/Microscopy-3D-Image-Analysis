# Reading .lif files

## General overview

This module defines tools useful for reading files readable by bioformats
library.

## Setup

To use bioformats you need to install package `openjdk-8-jdk` and set variable
`JAVA_HOME` to directory where it is installed.

LifReader and DenoisingLifReader should be used only with their `open` and
`close`
methods or with context manager. For example:
 ```(python)
reader = LifReader('path').open()
# Do something.
reader.close
```
or
```(python)
with LifReader('path') as reader:
    # Do something.
```
 When using either of those methods
virtual machine is automatically started and `javabridge.kill_vm` is scheduled
to be called when program exits (with atexit module).

## Using readers

General dataflow is defined in MetaReader interface in file `meta_readers.py`.
First metadata is read with `meta_reader` method. Then it is filtered with
`get_matching_meta`. Finally filtered metadata is used by `data_reader` to read
actual data. This is basically how `read_data_given_meta` method works.

## Using DenoisingLifReader

DenoisingLifReader is MetaReader for reading .lif files. It assumes that
metadata is in form of OME-XML and is
organized in following way. Each image in .lif file has name in form of
'sample_size_speed_region_slice' (e.g. 'Tonsil1_1024_600_FOV1_z1') and each
image has 16 channels. Consecutive channels have markers 'dapi', 
'pan-cytokeratin', 'ki67', 'cd3', 'dapi', and so on, also for first four
channels number of averaging steps is 8, next four 4, next four 2 and last four
1\. If some of these assumptions are not satisfied consider changing attributes
in `utils.py` file. Alternatively you can make another reader by inheriting
`MetaReader` class.

## Using LifReader

LifReader like DenoisingLifReader is used for reading .lif files, but with
slightly different organisation of metadata. It only assumes that metadata
is in form of OME-XML and it reads first channel's marker as 'dapi', second as
'pan-cytokeratin', third as 'ki67' and last one as 'cd3'. Recommended way to
use it is context manager.

## Using script to create directory tree

You can use `make_directories.py` script to read all images from all .lif files
in given directory and create directory tree with them. Each directory name
will describe parameters of images inside it. Also each image name will
describe its parameters. Exact layout of directories is described by 
`default_layout` in `make_directories.py`. For example if `default_layout` is
like this:
```(python)
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
```
for image with name 'Tonsil1_1024_600_FOV1_z1' directories 
`sample-Tonsil1_z1_region-FOV1_marker-dapi`, 
`sample-Tonsil1_z1_region-FOV1_marker-pan-cytokeratin`, 
`sample-Tonsil1_z1_region-FOV1_marker-ki67` and  
`sample-Tonsil1_z1_region-FOV1_marker-dapi` will be created. Each directory
will contain appropriate images, like 
`size-1024px_speed-600Hz_averaging_steps-8` (names will differ in number of
 averaging steps, see section above for explanation). Each OrderedDict in
 `default_layout` describes one level of directory tree. Last one describes
 files. For now to change this layout (order, form of information etc.) just
 edit `default_layout`. Script for creating directory tree can be called like
 this:
 ```(python)
 python -m clb.dataprep.lif.make_directories --load-dir <dir_with_lifs> --save-dir <dir_to_save_images> --ext <extension> [--resize]

```
File extension (`--ext`) should be in form '.ext', default is '.png'. 
With `--resize` flag each image will be upscaled to 2048x2048. 