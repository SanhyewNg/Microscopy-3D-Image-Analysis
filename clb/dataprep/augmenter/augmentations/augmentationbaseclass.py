from abc import abstractmethod
from random import choices


class AugmentationBaseClass:

    multichannel_img = False

    def __init__(self, probability, freeze_properties):
        """
        Args:
            probability (float): A probability of an augmentation being applied.
            freeze_properties (bool): A flag deciding if the properties of the
                augmentation should stay the same throughout multiple uses.
        """
        self.probability = probability
        self.freeze_properties = freeze_properties

    def augment(self, image):
        if self.check_if_augmenting():
            return self.apply_augmentation(image)
        else:
            return image

    def apply_augmentation(self, image):
        self._set_augmentation_properties(image)
        augmented_image = self._augment_image(image)
        if not self.freeze_properties:
            self.refresh_properties(image)

        return augmented_image

    def refresh_properties(self, image):
        self._reset_properties()
        self._set_augmentation_properties(image)

    def check_if_augmenting(self):
        return choices(population=[True, False],
                       weights=(self.probability, 1-self.probability))[0]

    @abstractmethod
    def _set_augmentation_properties(self, image):
        raise NotImplementedError

    @abstractmethod
    def _augment_image(self, image):
        raise NotImplementedError

    @abstractmethod
    def _reset_properties(self):
        raise NotImplementedError
