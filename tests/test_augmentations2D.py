import copy
import pytest
import unittest

import numpy as np
import numpy.testing as nptest

from clb.dataprep.augmentations2D import AugGenerator
from clb.dataprep.utils import rescale_to_float


@pytest.mark.preprocessing
class TestAugs(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        pass

    def test_augmentation_zero(self):
        image1 = np.random.random((200, 200))
        image2 = np.random.random((200, 200))

        augmentator = AugGenerator(mode='constant', seed=43)
        x, y = next(augmentator.flow([(image1, image2)], augs=0))

        nptest.assert_equal(image1, x)
        nptest.assert_equal(image2, y)

    def test_augmentation_one(self):
        image1 = (np.random.random((200, 200)) * 100).astype(np.uint8)
        image2 = (np.random.random((200, 200)) * 100).astype(np.uint8)

        augmentator = AugGenerator(mode='constant', seed=43)
        for idx, (x, y) in enumerate(augmentator.flow([(image1, image2)],
                                                      augs=1)):
            if idx == 0:
                nptest.assert_equal(image1, x)
                nptest.assert_equal(image2, y)
            else:
                self.assertFalse(np.array_equal(image1, x))
                self.assertFalse(np.array_equal(image2, y))

    def test_augmentation_transform_image_label_identical(self):
        image = (np.random.random((200, 200)) * 100).astype(np.uint8)
        label = copy.copy(image)
        nptest.assert_equal(image, label)

        augmentator = AugGenerator(mode='constant', seed=43)
        for x, y in augmentator.flow([(image, label)],
                                     augs=1,
                                     ensure_gt_binary=False):
            nptest.assert_equal(x, y)

    def test_augmentation_rgb_image(self):
        image = (np.random.random((256, 256, 3)) * 100).astype(np.uint8)
        label = (np.random.random((256, 256)) * 100).astype(np.uint8)

        augmentator = AugGenerator(mode='constant', seed=43)
        for x, y in augmentator.flow([(image, label)], augs=0):
            nptest.assert_equal(image, x)
            nptest.assert_equal(label, y)

    def test_augmentation_uint16_label_type_consistency(self):
        image = (np.random.random((256, 256, 3)) * 100).astype(np.uint8)
        label = (np.random.random((256, 256)) * 100).astype(np.uint16)

        augmentator = AugGenerator(mode='constant', seed=43)
        for _, y in augmentator.flow([(image, label)], augs=3):
            self.assertTrue(y.dtype == label.dtype)

    def test_augmentation_float_or_uint_roughly_invariant(self):
        image = (np.random.random((256, 256, 3)) * 100).astype(np.uint8)
        image_rescaled = rescale_to_float(image, 'float32')
        label = (np.random.random((256, 256)) * 100).astype(np.uint16)

        augmentator1 = AugGenerator(mode='constant', seed=43, enable_elastic=True)
        xs, _ = list(zip(*(augmentator1.flow([(image, label)], augs=5))))

        augmentator2 = AugGenerator(mode='constant', seed=43, enable_elastic=True)
        xs_from_rescaled, _ = list(zip(*(augmentator2.flow([(image_rescaled, label)], augs=5))))

        for x, x_from_rescaled in zip(xs, xs_from_rescaled):
            x_rescaled = rescale_to_float(x, 'float32')
            nptest.assert_almost_equal(x_rescaled, x_from_rescaled, decimal=2)

    def test_augmentation_uint16_img_type_consistency(self):
        image = (np.random.random((256, 256, 3)) * 100).astype(np.uint16)
        label = (np.random.random((256, 256)) * 100).astype(np.uint16)

        augmentator = AugGenerator(mode='constant', seed=43)
        for x, _ in augmentator.flow([(image, label)], augs=1):
            self.assertTrue(x.dtype == image.dtype)

    def test_augmentation_not_change_dimensions_reflect_mode(self):
        image = (np.random.random((256, 256, 3)) * 100).astype(np.uint8)
        label = (np.random.random((256, 256)) * 100).astype(np.uint16)

        augmentator = AugGenerator(pad=256, mode='reflect', seed=43)
        for x, y in augmentator.flow([(image, label)], augs=1):
            nptest.assert_equal(x.shape, (256, 256, 3))
            nptest.assert_equal(y.shape, (256, 256))

    def test_augmentation_not_change_dimensions_constant_mode(self):
        image = (np.random.random((256, 256, 3)) * 100).astype(np.uint8)
        label = (np.random.random((256, 256)) * 100).astype(np.uint16)

        augmentator = AugGenerator(mode='constant', seed=43)
        for x, y in augmentator.flow([(image, label)], augs=1):
            nptest.assert_equal(x.shape, (256, 256, 3))
            nptest.assert_equal(y.shape, (256, 256))

    def test_label_binarized_when_ensure_gt_binary(self):
        label = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        augmentator = AugGenerator(mode='constant', seed=43)

        bin_label = augmentator.ensure_gt_binary(label, thresh=0.5,
                                                 min_val=0, max_val=1)

        self.assertTrue(len(np.unique(bin_label) == 2))

    def test_label_binary_after_augmentation(self):
        image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        label = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        augmentator = AugGenerator(mode='constant', seed=43)
        for idx, (_, y) in enumerate(augmentator.flow([(image, label)],
                                                      augs=3)):
            if idx > 0:
                self.assertTrue(len(np.unique(y)) == 2)

    def test_multiple_augmentations_same_results_when_seeded(self):
        image = np.random.randint(0, 256, (256, 256))
        label = np.random.randint(0, 256, (256, 256))

        # Creating seeding AugGenerator() object here - in it's __init__
        # method imgaug.seed() function is called. Note, that this seeds
        # GLOBAL random state of the whole module, not a random state of the
        # imgaug Sequence object or anything like this. Thus, when trying to
        # test whether seeded augmenters are reproducable it's essential to
        # keep in mind, that calling augmentator1.flow() function changes the
        # global random state of imgaug module - it calls
        # seq.to_deterministic() method, which somehow replaces the global
        # state by the local one.
        augmentator1 = AugGenerator(mode='constant', seed=43)

        x_gen1, y_gen1 = [], []
        x_gen2, y_gen2 = [], []

        for x, y in augmentator1.flow([(image, label)], augs=1):
            x_gen1.append(x)
            y_gen1.append(y)

        # Create second augmentator here and reseed the whole module in it's
        # __init__ method - it replaces imgaug global random state modified by
        # previous augmenter to a desired new seed.
        augmentator2 = AugGenerator(mode='constant', seed=43)
        for x, y in augmentator2.flow([(image, label)], augs=1):
            x_gen2.append(x)
            y_gen2.append(y)

        for x1, x2 in zip(x_gen1, x_gen2):
            nptest.assert_equal(x1, x2)

        for y1, y2 in zip(y_gen1, y_gen2):
            nptest.assert_equal(y1, y2)

    def tearDown(self):
        pass
