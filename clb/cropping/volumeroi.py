import random
from copy import copy

import numpy as np
import yaml

from clb.cropping import CropInfo
from clb.utils import channels_count


class VolumeROI:
    """
    Represent 3D ROI inside a bigger volume. Consists of:
    - numpy array of roi shape
    - CropInfos that express where this numpy array is original volume.
    """

    def __init__(self, crop_infos, volume):
        """
        Combine CropInfos and actual numpy array into a VolumeRoi
        Args:
            crop_infos: non empty list of CropInfos that form a 3D block
            volume: Z x Y x X or Z x Y x X x C numpy array that comply with crop_infos shape
        """
        assert len(crop_infos) > 0, "Empty list of crop info for VolumeROI creation."
        assert CropInfo.block_size(crop_infos)[:3] == volume.shape[:3]
        self.crop_infos = crop_infos
        self.crop_volume = volume
        self.zyx_start = crop_infos[0].zyx_start

    @property
    def xs_span(self):
        return self.zyx_start[2], self.zyx_start[2] + self.shape[2]

    @property
    def ys_span(self):
        return self.zyx_start[1], self.zyx_start[1] + self.shape[1]

    @property
    def zs_span(self):
        return self.zyx_start[0], self.zyx_start[0] + self.shape[0]

    @property
    def span(self):
        return [self.zs_span, self.ys_span, self.xs_span]

    @property
    def shape(self):
        return self.crop_volume.shape

    @property
    def dtype(self):
        return self.crop_volume.dtype

    @staticmethod
    def create_empty(crop_infos, dtype, channels=None):
        """
        Create VolumeROI based on CropInfos filled with zeros
        Args:
            crop_infos: non empty list of CropInfos that form a 3D block
            dtype: type of the numpy array inside VolumeROI
            channels: number of channels to create or None if 3D numpy array should be used

        Returns:
            VolumeROI filled with zeroes.
        """
        shape = CropInfo.block_size(crop_infos)
        if channels is not None:
            shape = shape + (channels,)
        crop_volume = np.zeros(shape, dtype=dtype)

        return VolumeROI(crop_infos, crop_volume)

    @staticmethod
    def from_absolute_crop(crop_infos, absolute_volume):
        """
        Define VolumeROI from original volume based on CropInfos.
        CropInfos are first confined to absolute_volume bounds.
        Args:
            crop_infos: non empty list of CropInfos that form a 3D block representing ROI
            absolute_volume:  Z x Y x X or Z x Y x X x C numpy array showing the original volume

        Returns:
            VolumeROI representing ROI restricted so that it is inside the original volume.
        """
        bounded_crop_infos = CropInfo.restrict_infos(crop_infos, absolute_volume.shape[:3])
        cropped_volume = CropInfo.crop_volume(absolute_volume, bounded_crop_infos)
        return VolumeROI(bounded_crop_infos, cropped_volume)

    @staticmethod
    def from_absolute_crop_with_padding(crop_infos, absolute_volume):
        """
        Define VolumeROI from original volume based on CropInfos.
        CropInfos not confined to absolute_volume bounds the volume is padded instead.
        Args:
            crop_infos: non empty list of CropInfos that form a 3D block representing ROI
            absolute_volume:  Z x Y x X or Z x Y x X x C numpy array showing the original volume

        Returns:
            VolumeROI representing ROI padded so that all CropInfos are in bounds of the new volume
        """
        bounded_roi = VolumeROI.from_absolute_crop(crop_infos, absolute_volume)
        padded_roi = VolumeROI.create_empty(crop_infos, dtype=bounded_roi.dtype,
                                            channels=channels_count(absolute_volume))
        padded_roi.implant(bounded_roi)
        return padded_roi

    def span_intesect(self, volume_roi, axis):
        """
        Determine the intersection of this VolumeROI with the provided on the given axis.
        Args:
            volume_roi: other VolumeROI to intersect with
            axis: on which to calculate intersection

        Returns:
            None if no intersection
                or (start, end) representing the intersection on 1D axis.

        """
        my_span = np.array(self.span[axis])
        other_span = np.array(volume_roi.span[axis])

        intersect = max(my_span[0], other_span[0]), min(my_span[1], other_span[1])
        intersect_size = max(0, intersect[1] - intersect[0])

        if intersect_size == 0:
            intersect = None

        return intersect

    def slice_from_absolute(self, axis, absolute_span):
        """
        Prepare slice on the given axis of numpy array of this VolumeROI representing some absolute interval.
        Args:
            axis: on which to determine slice
            absolute_span: (start, end) span to slice using absolute coordinates

        Returns:
            slice which and be used to crop from ROI based on absolute coordinated
        """
        my_span = np.array(self.span[axis])

        start_relative = max(absolute_span[0] - my_span[0], 0)
        end_relative = max(absolute_span[1] - my_span[0], 0)

        return slice(start_relative, end_relative)

    def extract_absolute(self, absolute_spans_zyx):
        """
        Crop volume from ROI that intersects with provided absolute spans.
        Args:
            absolute_spans_zyx: spans in original volume coordinates to crop

        Returns:
            numpy volume representing the intersecting part of ROI and provided spans
                Note that this can have size that do not comply with spans.
        """

        relative_slices = tuple(self.slice_from_absolute(axis, span) for axis, span in enumerate(absolute_spans_zyx))
        return self.crop_volume[relative_slices]

    def implant(self, volume_roi):
        """
        Copy the content of another VolumeROI into this one.
        Args:
            volume_roi: ROI that we want to implant into this one.
        """
        intesected_spans = [self.span_intesect(volume_roi, axis) for axis, _ in enumerate(self.span)]
        if None in intesected_spans:
            return None

        my_slices = tuple([self.slice_from_absolute(axis, span) for axis, span in enumerate(intesected_spans)])

        self.crop_volume[my_slices] = volume_roi.extract_absolute(intesected_spans)

    def __str__(self):
        return "VolumeROI: Shape={}, Span={}".format(self.shape, self.span)
