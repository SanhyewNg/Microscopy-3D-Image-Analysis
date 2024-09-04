import numpy as np
import skimage.transform


class MultiTransformer:
    def __init__(self, use_rotation, use_flips):
        self.rotations = [0, 90, 180, 270] if use_rotation else [0]
        if use_flips and use_rotation:
            # some combinations of flips and rotations are equivalent
            self.rotations = [0, 90]
        self.use_flips = use_flips
        self.number_of_trans = len(self.rotations)
        if use_flips:
            self.number_of_trans *= 4

    def generate_transformations(self, images):
        for im in images:
            for rotation in self.rotations:
                rotated = skimage.transform.rotate(im, rotation)

                yield rotated
                if self.use_flips:
                    flip_vertical = np.flip(rotated, 0)
                    yield flip_vertical

                    flip_horizontal = np.flip(rotated, 1)
                    yield flip_horizontal

                    flip_both = np.flip(flip_vertical, 1)
                    yield flip_both

    def merge_transformations(self, images):
        # use median to merge self.number_of_trans images
        chunks = np.array_split(np.array(images), len(images) / self.number_of_trans)
        for chunk in chunks:
            cur = 0
            for rotation in self.rotations:
                for i in range(0, 4):
                    if cur % 4 == 1:
                        chunk[cur] = np.flip(chunk[cur], 0)
                    if cur % 4 == 2:
                        chunk[cur] = np.flip(chunk[cur], 1)
                    if cur % 4 == 3:
                        chunk[cur] = np.flip(chunk[cur], 1)
                        chunk[cur] = np.flip(chunk[cur], 0)

                    chunk[cur] = skimage.transform.rotate(chunk[cur], -rotation)
                    cur += 1

            yield np.median(chunk, 0)
