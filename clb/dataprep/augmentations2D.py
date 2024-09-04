from copy import deepcopy

import cv2
import imgaug
import numpy as np
from imgaug import augmenters

from vendor.genny.genny.wrappers import obj_gen_wrapper


class AugGenerator:

    def __init__(self, mode='constant', enable_elastic=True, pad=None,
                 seed=None):
        """
        Args:
            pad: should be equal to dim of the image (squarish)
        """

        self.mode = mode
        self.pad = pad

        if self.mode == 'reflect' and self.pad is None:
            raise ValueError('reflect mode requires pad to be given')

        # Initialize imgaug seed for reproducibility.
        imgaug.seed(seed)

        augmentations = [
            augmenters.Fliplr(0.5),
            augmenters.Flipud(0.5),

            self.sometimes(augmenters.Affine(scale=(0.8, 1.2))),
            self.sometimes(augmenters.Affine(translate_percent=(-0.2, 0.2))),
            self.sometimes(augmenters.Affine(rotate=(0, 359))),
            self.sometimes(augmenters.Affine(shear=(-10, 10))),
        ]

        if enable_elastic:
            augmentations.append(
                augmenters.Sometimes(0.8,
                                     augmenters.PiecewiseAffine(scale=(0.02,
                                                                       0.04)))
            )

        if self.mode == "reflect":
            augmentations.append(
                augmenters.Crop(px=self.pad,
                                keep_size=False,
                                random_state=seed)
            )

        self.seq = augmenters.Sequential(augmentations)

    def sometimes(self, aug, p=0.5):
        """Wrapper for imgaug `Sometimes()` method.

        Args:
            aug: augmentation to be executed sometimes
            p: probability of the augmentation being applied
        """
        return augmenters.Sometimes(p, aug)

    def make_border(self, array):
        """Wrapper around openCV copyMakeBorder function with dimension fix.

        Args:
            array:  input array that will be padded (height, width) or
                    (height, width, channels)

        Returns:
            padded array: (2 * height, 2 * width) or
                          (2 * height, 2 * width, channels)
        """
        pad = self.pad

        padded_array = cv2.copyMakeBorder(array, pad, pad, pad, pad,
                                          cv2.BORDER_REFLECT)

        return padded_array

    def ensure_gt_binary(self, gt, min_val, max_val, thresh=0.5):
        """Ensure ground truth contains values from set {min_val, max_val}.

        It is important to ensure final gt is binary since augmentations
        like elastic distortions perform some gaussian blurring inside them,
        thus crashing gt's binary nature. After running such operations, we
        want to ensure that final labels are binary again. If gt is
        multi-channel, each channel will be ensured to have values from
        aforementioned set.

        Args:
            gt: ground truth that will be checked (must be of integer type)
            thresh: values above (max value of dtype * `thresh`) will have
                    `max_val` assigned to them. Values below (max value of
                    dtype * `thresh`) will have `min_value` assugned to
                    them.
            min_val: value meaning 'nothing interesting here'
            max_val: value meaning 'something interesting here'

        Returns:
            modified ground truth with values converted into specific values.

        Raises:
            ValueError, when `gt` is not of integer type.
        """
        bin_gt = deepcopy(gt)
        type_max = np.iinfo(gt.dtype).max
        thresh_value = np.ceil(type_max * thresh)

        for channel in range(bin_gt.shape[-1]):
            pixels_above_thresh = bin_gt[..., channel] > thresh_value
            pixels_below_eq_thresh = bin_gt[..., channel] <= thresh_value

            bin_gt[pixels_above_thresh, channel] = max_val
            bin_gt[pixels_below_eq_thresh, channel] = min_val

        return bin_gt

    @obj_gen_wrapper
    def flow(self, data_gen, augs=0, ensure_gt_binary=True, bin_min_value=0,
             bin_max_value=255, bin_thresh=0.5):
        """Produce augmentation results.

        Args:
            data_gen: generator yielding tuples (image, ground truth)
            augs: number of augmented images that will be generated
            ensure_gt_binary: after augmentation process (that spoils binary
                              nature of the label due to heavy elastic
                              distortions applied) it is necessary to
                              restore binary nature of the label). This
                              parameter allows to turn it on / off.
            bin_min_value: if `ensure_gt_binary=True`, then during
                           binarization process this value is used for pixels
                           with background.
            bin_max_value: if `ensure_gt_binary=True`, then during
                           binarization process this value is used for pixels
                           with objects of interests.

        Yields:
            tuple (image, ground truth)
        """
        for img, gt in data_gen:

            # First yielded tuple contains non-augmented data.
            yield deepcopy(img), deepcopy(gt)

            if augs == 0:
                continue

            # Increase image dimensions to ensure there will be NO black areas
            # as a side effect of augmenting original image. Image is cropped
            # at the end of augmentation sequence back to the original shape.
            if self.mode == 'reflect':
                img = self.make_border(img)
                gt = self.make_border(gt)

            for _ in range(augs):
                # Convert augmentator to be deterministic in order to apply
                # exactly the same transformations both to images and labels.
                seq_det = self.seq.to_deterministic()

                aug_img = seq_det.augment_images([img])[0]
                aug_gt = seq_det.augment_images([gt])[0]

                if ensure_gt_binary:
                    aug_gt = self.ensure_gt_binary(gt=aug_gt,
                                                   min_val=bin_min_value,
                                                   max_val=bin_max_value,
                                                   thresh=bin_thresh)

                yield aug_img, aug_gt
