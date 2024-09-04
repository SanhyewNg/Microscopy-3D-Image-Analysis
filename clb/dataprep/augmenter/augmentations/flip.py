from random import randint

import numpy as np

from clb.dataprep.augmenter.augmentations.augmentationbaseclass import AugmentationBaseClass


class Flip(AugmentationBaseClass):

    def __init__(self, probability=1.0, axis=None, freeze_properties=False):
        self.axis = axis
        super().__init__(probability=probability,
                         freeze_properties=freeze_properties)

    def _set_augmentation_properties(self, image=None):
        if self.axis is None:
            self._set_axis(len(image.shape))

    def _set_axis(self, number_of_dims):
        if self.multichannel_img:
            number_of_dims -= 1
        self.axis = randint(0, number_of_dims - 1)

    def _augment_image(self, image):
        augmented_image = np.flip(image, self.axis)
        return augmented_image

    def _reset_properties(self):
        self.axis = None
