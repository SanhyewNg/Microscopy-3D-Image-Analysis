from random import randint, choices

import scipy.ndimage.interpolation as sni

from clb.dataprep.augmenter.augmentations.augmentationbaseclass import AugmentationBaseClass


class Shift(AugmentationBaseClass):

    shift = 0

    def __init__(self, shift_range, probability=1.0, axes=None,
                 freeze_properties=False):
        self.axes = axes
        self.shift_range = shift_range
        super().__init__(probability=probability,
                         freeze_properties=freeze_properties)

    def _set_augmentation_properties(self, image):
        shift_axes = self._set_augmentation_axes(image.shape)
        self._set_shift(shift_axes)

    def _augment_image(self, image):
        return sni.shift(input=image, shift=self.shift)

    def _set_augmentation_axes(self, image_shape):
        number_of_axes = len(image_shape)
        if self.multichannel_img:
            number_of_axes -= 1
        if self.axes is None:
            shift_axes = choices([True, False], k=number_of_axes)
        else:
            shift_axes = [False, ] * number_of_axes
            for axis in self.axes:
                shift_axes[axis] = True
        return shift_axes

    def _set_shift(self, shift_axes):
        shift_value = randint(self.shift_range[0], self.shift_range[1])
        self.shift = (i * shift_value for i in shift_axes)

    def _reset_properties(self):
        self.axes = None
