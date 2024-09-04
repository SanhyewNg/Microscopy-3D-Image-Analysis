# Augmentations module

This is an augmentation module written for the purpose of augmenting 3D nuclei
cubes. It's modular, so if there is a need you can add more augmentations.
The ones that are already there has been tested empirically, but due to the 
sudden project cancellation there are no test or prediction data to lean on.
This is where we suggest to start - add some test for checking if parameters are 
passed properly to the functions and train the exisiting vgg on the augmented data.

Side note: These modules work on 3D as well as 2D, so if there is a need to
replace imgaug entirely, you can use this.