from random import sample, uniform

import scipy.ndimage.interpolation as sni

from clb.dataprep.augmenter.augmentations.augmentationbaseclass import AugmentationBaseClass


class Rotation(AugmentationBaseClass):

    def __init__(self, angle_range, probability=1.0, axes=None, angle=None,
                 freeze_properties=False):
        self.angle_range = angle_range
        self.axes = axes
        self.angle = angle
        super().__init__(probability=probability,
                         freeze_properties=freeze_properties)

    def _set_augmentation_properties(self, image):
        if self.axes is None:
            self._set_augmentation_axes(len(image.shape))
        if self.angle is None:
            self.angle = uniform(self.angle_range[0], self.angle_range[1])

    def _set_augmentation_axes(self, number_of_axes):
        if self.multichannel_img:
            number_of_axes -= 1
        self.axes = sample(range(number_of_axes), k=2)

    def _augment_image(self, image):
        return sni.rotate(input=image, angle=self.angle,
                          axes=self.axes, reshape=False)

    def _reset_properties(self):
        self.axes = None
        self.angle = None
