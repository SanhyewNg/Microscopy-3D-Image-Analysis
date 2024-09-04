import math
from random import uniform

import numpy as np
import scipy.ndimage.interpolation as sni

from clb.dataprep.augmenter.augmentations.augmentationbaseclass import AugmentationBaseClass


class Scale(AugmentationBaseClass):

    scale_factors = None

    def __init__(self, scale_range, probability=1.0, axes=None,
                 freeze_properties=False):
        self.scale_range = scale_range
        self.axes = axes
        super().__init__(probability, freeze_properties)

    def _set_augmentation_properties(self, image=None):
        scale_axes = self._set_augmentation_axes(len(image.shape))
        self._set_scale_factors(scale_axes)

    def _augment_image(self, image):
        scaled_image = sni.zoom(input=image, zoom=self.scale_factors)
        scaled_image = self.adjust_zoomed_size(scaled_image, image.shape)

        return scaled_image

    def adjust_zoomed_size(self, zoomed_image, image_shape):
        for axis, factor in enumerate(self.scale_factors):
            if factor > 1.0:
                zoomed_image = self.clip_after_zoom(zoomed_image, image_shape,
                                                    axis)
            elif factor < 1.0:
                zoomed_image = self.pad_after_zoom(zoomed_image, image_shape,
                                                   axis)
        return zoomed_image

    def clip_after_zoom(self, zoomed_image, image_shape, axis):
        clipped_dimensions = []
        for current_axis, dimensions in \
                enumerate(zip(image_shape, zoomed_image.shape)):
            clipped_dimensions.append(
                self.set_clipping_indices(axis, current_axis, dimensions[0],
                                          dimensions[1]))

        clipped_image = zoomed_image[tuple(clipped_dimensions)]

        return clipped_image

    def pad_after_zoom(self, zoomed_image, image_shape, axis):
        padded_dimensions = []
        for current_axis, dimensions in \
                enumerate(zip(image_shape, zoomed_image.shape)):
            padded_dimensions.append(
                self.set_padding_indices(axis, current_axis, dimensions[0],
                                         dimensions[1]))
        return np.pad(zoomed_image, padded_dimensions, 'constant')

    def _set_augmentation_axes(self, number_of_axes):
        if self.multichannel_img:
            number_of_axes -= 1
        if self.axes is None:
            scale_axes = [True, ] * number_of_axes
        else:
            scale_axes = [False, ] * number_of_axes
            for axis in self.axes:
                scale_axes[axis] = True
        return scale_axes

    def _set_scale_factors(self, scale_axes):
        scale_value = uniform(self.scale_range[0], self.scale_range[1])
        self.scale_factors = [axis * scale_value for axis in scale_axes]

    def set_clipping_indices(self, axis, current_axis, image_dimension,
                             zoom_dimension):
        if current_axis == axis:
            return self.calculate_clipping_indices(image_dimension,
                                                   zoom_dimension)
        else:
            return slice(0, zoom_dimension)

    def set_padding_indices(self, axis, current_axis, image_dimension,
                            zoom_dimension):
        if current_axis == axis:
            return self.calculate_padding_dimensions(image_dimension,
                                                     zoom_dimension)
        else:
            return 0, 0

    @staticmethod
    def calculate_clipping_indices(image_dimension, zoom_dimension):
        clip_min = zoom_dimension // 2 - math.ceil(image_dimension / 2)
        clip_max = zoom_dimension // 2 + math.floor(image_dimension / 2)

        return slice(int(clip_min-1), int(clip_max-1))

    @staticmethod
    def calculate_padding_dimensions(image_dimension, zoom_dimension):
        pad_min = image_dimension // 2 - math.floor(zoom_dimension / 2)
        pad_max = image_dimension - (image_dimension // 2 +
                                     math.ceil(zoom_dimension / 2))

        return int(pad_min), int(pad_max)

    def _reset_properties(self):
        self.axes = None
        self.scale_factors = None
