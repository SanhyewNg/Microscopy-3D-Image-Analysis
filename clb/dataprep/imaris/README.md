# H5PY wrapper for writing into IMARIS format

The goal of this wrapper is to recreate unique imaris file structure, while 
using h5py module methods. It is possible due to the fact that .ims is an 
open format based on hdf5, which this module is for.

## Imaris file format

Imaris file format has a characteristic structure, available here:
http://open.bitplane.com/Default.aspx?tabid=268

The goal of this wrapper is to recreate this structure, while keeping it
available for easy modification and adding of data.

## General idea

Instantiating an ImsFile class object causes its constructor and the 
constructors of subsequent subclasses to create a template imaris file,
ready for taking data in a form of numpy arrays. Each constructor calls
create_group method from h5py module upon Group object created in its parent,
going all the way up to the root directory of the file. Additionally, some of 
the groups need to have their attributes specified [see Imaris documentation 
above for more info.], which is done by attrs field of a dictionary type.
This field is passed with the corresponding group to the global update_attrs 
method, which is later useful for modification of the said attributes when 
the need arises.

## Specific data shape and formats

Attributes need to be passed as one-byte encoded string ending on a null-byte.
(dtype="|S1" in Numpy). 

Histograms for datasets always have 256 bins, the last bin is corresponding to 
the maximum value of the particular dataset dtype and minimum value corresponds 
to the minimum value of the dataset dtype. Histogram is 64bit.

## Adding data to initialized ims file

By using add_channel method on ImsFile class and passing a numpy array
with suitable arguments and metadata, the wrapper is able to accomodate 
the file structure to the freshly passed data. The idea is that each parent 
class passes the data to its children with suitable attributes and finally
the data is passed to the ims file by an object of the Channel class.
The data are NOT STORED in the wrapper objects, but what the attributes and
metadata are.

## Color modes

#### BaseColor

Uses basic RGB format, Color attribute in DataSetInfo/Channel denotes 
which color is being used (1.000 0.000 0.000 - red, 0.000 1.000 0.000 - blue and 
0.000 0.000 1.000 -green). Pixel value corresponds to its intensity in the 
channel.

#### TableColor

Loads a ColorTable (saved in the colortable.yaml in this module) containing 256 
colours that are loaded into ColorTable parameter as a string. Each color is 
described by three floats ranging from 0 to 1, which denote intensity in the 
corresponding RGB channels. ColorRange attribute takes in 2 values which are
min and max pixel values. What follows is that each pixel value has one of the
table colours assigned to them, which allows to visualise instance segmentation 
in Imaris

## Cell Objects

TBD

## Reading and writing data to ims file

It's possible to extract some channels from ims file with imsmanip.py script.
There is also possibility to add channel to ims file from tif file.

To extract channels use:
```
python -m clb.scripts.imsmanip extract --ims-path <path_to_ims> \
--tif-path <path_to_tif> [--channels <channels>] 
```
`<channels>` should be comma separated list of channels (just commas, without
spaces). To extract all channels just omit this argument.

To add channel to existing or new ims file use:
```
python -m clb.scripts.imsmanip add --ims-path <path_to_ims> \
--tif-path <path_to_tif> --channel <channel> --color <color> --name <name>
```

`<color>` is channel color in imaris in form of a string (e.g. 'Red', 'Green').
`<name>` is channel name also in form of a string.