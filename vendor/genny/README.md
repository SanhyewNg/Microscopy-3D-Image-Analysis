# genny - convenient data generation

This module is intended to serve as a wrapper around generators. It makes it possible to chain series of generators in a clear, convenient way. This functionality can be especially useful when your pipeline for creating input data consists of multiple generators, e.g.:

```python
from [your module name].genny.genny.wrappers import gen_wrapper
import cv2

@gen_wrapper
def filenames_generator(images_dir, masks_dir):
    images_files = sorted(glob(images_dir))
    masks_files = sorted(glob(masks_dir))

    while True:
        for image_file, mask_file in zip(images_files, masks_files):
            yield image_file, mask_file


@gen_wrapper
def raw_data_generator(filenames_gen):

    for image_file, mask_file in filenames_gen:
        img = cv2.imread(image_file)
        mask = cv2.imread(mask_file)

        yield img, mask


@gen_wrapper
def invert_generator(data_gen):
    for img, mask in data_gen:
        inverted_img = cv2.bitwise_not(img)
        yield inverted_img, mask


if __name__ == '__main__':
    chained_generators = (
        filenames_generator(images_dir='dummy/dir', masks_dir='dummy_dir') |
        raw_data_generator() | # Note that you don't pass first argument - previous generator.
        invert_generator()
    )

    for img, mask in chained_generators():
        # Do your stuff here.

```

It is also possible to wrap generators that are part of some class, e.g.:

```python
from [your module name].genny.genny.wrappers import gen_wrapper, obj_gen_wrapper

class Augmentator():
    # ...
    # Some other methods and parameters here
    # ...

    @obj_gen_wrapper
    def augment(self, data_gen, ...):
        for img, mask in data_gen:
            # Your augmentations go here
            yield augmented_img, augmented_mask


@gen_wrapper
def filenames_generator(images_dir, masks_dir):
    images_files = sorted(glob(images_dir))
    masks_files = sorted(glob(masks_dir))

    while True:
        for image_file, mask_file in zip(images_files, masks_files):
            yield image_file, mask_file


@gen_wrapper
def raw_data_generator(filenames_gen):

    for image_file, mask_file in filenames_gen:
        img = cv2.imread(image_file)
        mask = cv2.imread(mask_file)

        yield img, mask


if __name__ == '__main__':
    augmentator = Augmentator()

    chained_generators = (
        filenames_generator(images_dir='dummy/dir', masks_dir='dummy_dir') |
        raw_data_generator() | # Note that you don't pass first argument - previous generator.
        augmentator.augment()
    )

    for img, mask in chained_generators():
        # Do your stuff here.

```
