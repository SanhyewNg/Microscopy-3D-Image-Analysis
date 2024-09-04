from copy import deepcopy

from clb.dataprep.augmenter.augmentations import *
from .augmentations.augmentationbaseclass import AugmentationBaseClass
from vendor.genny.genny.wrappers import obj_gen_wrapper


class Augmenter:

    def __init__(self, augmentations, multichannel_img=False, freeze=False):
        self.augmentations = augmentations
        self.freeze = freeze
        for augmentation in augmentations:
            if not isinstance(augmentation, AugmentationBaseClass):
                raise Exception(str(augmentation) + " is not a "
                                                    "valid augmentation")
            if multichannel_img:
                augmentation.multichannel_img = True

    def freeze_properties(self, freeze=False):
        self.freeze = freeze
        for augmentation in self.augmentations:
            augmentation.freeze_properties = freeze

    def set_all_probabilities_to_value(self, probability):
        for augmentation in self.augmentations:
            augmentation.probability = probability

    def augment(self, image, ground_truth=None):
        for augmentation in self.augmentations:
            if ground_truth is None:
                image = augmentation.augment(image)
                return image
            else:
                image, ground_truth = \
                    self.__augment_image_pair(augmentation, image, ground_truth)
                return image, ground_truth

    def __augment_image_pair(self, augmentation, image, ground_truth):
        image = augmentation.augment(image)
        augmentation.freeze_properties = True
        ground_truth = augmentation.augment(ground_truth)
        if not self.freeze:
            augmentation.freeze_properties = False

        return image, ground_truth


@obj_gen_wrapper
def augmentations_generator(data_gen, number_of_augmentations=0):
    """Produce augmentation results.
    Args:
        data_gen: generator yielding tuples (image, ground truth).
        number_of_augmentations: number of augmented images that
            will be generated.

    Yields:
        tuple (image, ground truth)
    """
    augmentations = [Flip(),
                     Rotation(angle_range=(0.0, 359.0)),
                     Scale(scale_range=(0.8, 1.2)),
                     Shift(0.5, 20)]
    augmenter = Augmenter(augmentations=augmentations,
                          freeze=False)
    augmenter.set_all_probabilities_to_value(0.5)
    for img, gt in data_gen:

        yield img, gt

        if number_of_augmentations == 0:
            continue

        for _ in range(number_of_augmentations):
            augmented_img, augmented_gt = augmenter.augment(image=img,
                                                            ground_truth=gt)
            yield augmented_img, augmented_gt


if __name__ == "__main__":
    aug_gen = augmentations_generator()
