# Recommended way to read data

Recommended way to read data is using `clb.dataprep.get_volume_reader` 
function. It returns `VolumeIter` object. This object can be indexed as two
dimensional array, where first axis is z and second axis is c. Indexing 
`VolumeIter` returns another `VolumeIter` with only selected indices. To 
convert `VolumeIter` into numpy array use `.to_numpy` method or `np.asarray`.
Iterating through `VolumeIter` gives consecutive slices as numpy arrays.

# Imageio plugins
Imageio has four modes and four corresponding functions for reading images:
* imread for mode 'i'
* mimread for mode 'I'
* volread for mode 'v'
* mvolread for mode 'V'

So for example if we add plugin for .lif files with mode 'I', when someone
tries to read .lif file with that plugin with `volread` imageio raises 
exception. Only reading with `mimread` is allowed.

# Plugins used in projects
Currently we have plugins for .tif and .lif files and both can be used
with 'v' mode.

# What to use and how
First of all there is a possibility to use `imageio.get_reader` function, which
is the most flexible way of using plugins. This function returns suitable
reader, which is an iterable of slices in file (or series in case of .lif
file). It also has length. After using reader this way you should probably call
`reader.close()`. Alternatively you can use context manager, for example:
```python
with imageio.get_reader('file.lif') as reader:
    print(len(reader)) # Number of slices.
    stack = list(reader) # Whole stack.
```
You can pass additional arguments to `get_reader`, for example channel and
series (for .lif file).

Second option is to use `imageio.volread`. It will return squezeed image, so
if you want to have additional axis even if there is only one slice, you have
to add it yourself.

There are also functions like `load_tiff_stack` and `read_volume` in
`clb.dataprep.utils`. Second one is an extension of the first one on .lif
files. They are used in one particular case. There are multichannel .tif files
in project, where actual data is only in one channel. When those functions
are used, they merge channels of such a file by taking maximum along channels.
In case of .lif files they just read specified channel.
