# Tests

Tests should be run with pytest from the project root directory. Because of the size of the 
repository tests has been tagged with pytest marks. Here's a list of currently used tags:
```
classification
hypertuning
integration
io
preprocessing
postprocessing
segmentation
statistics
```
In order to use specific tags user needs to define them with `-m` flag.
Example command:

`pytest -m classification tests/`

Several tags can be used as well as excluding other tags by using simple logical operators. All tags needs to be in one string.
Example command:

`pytest -m "classification and not integration"`