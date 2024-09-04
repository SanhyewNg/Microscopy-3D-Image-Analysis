import argparse
import collections
import math

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

from clb.dataprep.utils import reduce_to_max_channel

ImageSlice = collections.namedtuple('ImageSlice', ['img', 'x', 'y'])


class ImageSlicer:

    def __init__(self, crop_width, crop_height, pad):
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.padding = pad
        self.input_shape = ()

    def divide_image(self, img):
        self.input_shape = img.shape

        # Increase image size to crop it correctly (use mirror reflection along
        # edges of the image).
        padded_img = cv2.copyMakeBorder(img, self.padding, self.padding,
                                        self.padding, self.padding,
                                        cv2.BORDER_REFLECT)

        img_h, img_w = img.shape[:2]

        imgs_per_row = img_w // self.crop_width
        imgs_per_col = img_h // self.crop_height

        if img_w % self.crop_width > 0:
            imgs_per_row += 1

        if img_h % self.crop_height > 0:
            imgs_per_col += 1

        res = []
        for row in range(imgs_per_col):
            for col in range(imgs_per_row):
                start_x = col * self.crop_width + self.padding
                end_x = start_x + self.crop_width

                if end_x > self.padding + img_w:
                    end_x = self.padding + img_w
                    start_x = end_x - self.crop_width

                start_y = row * self.crop_height + self.padding
                end_y = start_y + self.crop_height

                if end_y > self.padding + img_h:
                    end_y = self.padding + img_h
                    start_y = end_y - self.crop_height

                padded_start_x = start_x - self.padding
                padded_start_y = start_y - self.padding
                padded_end_x = end_x + self.padding
                padded_end_y = end_y + self.padding

                area = slice(padded_start_y, padded_end_y), slice(padded_start_x, padded_end_x)
                cropped = padded_img[area]

                res.append(ImageSlice(cropped, col, row))
        return res

    def remove_padding(self, image):
        img_h = image.shape[0] - 2 * self.padding
        img_w = image.shape[1] - 2 * self.padding
        area_to_crop = [slice(self.padding, self.padding + img_h),
                        slice(self.padding, self.padding + img_w)]

        return image[area_to_crop]

    def stitch_images(self, images):
        sample_img = images[0].img
        img_h, img_w = np.array(sample_img.shape[:2]) - 2 * self.padding

        mosaic_shape = self.input_shape[:2] + sample_img.shape[2:]
        mosaic_y_size, mosaic_x_size = self.input_shape[:2]
        mosaic = np.zeros(mosaic_shape, dtype=sample_img.dtype)

        for img in images:
            start_x = img.x * img_w
            start_y = img.y * img_h

            if start_x + img_w > mosaic_x_size:
                start_x = mosaic_x_size - img_w

            if start_y + img_h > mosaic_y_size:
                start_y = mosaic_y_size - img_h

            tile = self.remove_padding(img.img)
            area_to_paste = [slice(start_y, start_y + img_h),
                             slice(start_x, start_x + img_w)]

            # Extend slice if array is 3d.
            if len(tile.shape) > 2:
                area_to_paste.append(slice(0, tile.shape[2]))
            mosaic[area_to_paste] = tile

        return mosaic


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image slicer.',
                                     add_help=False)

    required = parser.add_argument_group('required arguments')
    required.add_argument('--image', help='2d input TIFF path')

    return parser.parse_args()


def main():
    ###################################
    # Sample usage of ImageSlicer class
    ###################################
    args = parse_arguments()
    crop_dim = 196

    # Split image into a tiles of a given size and add padding.
    slicer = ImageSlicer(crop_dim, crop_dim, 30)

    image = imageio.imread(args.image)
    image = reduce_to_max_channel(image)
    image_tiles = slicer.divide_image(image)

    # Run segmentation separately on all of tiles.
    # ...
    # [your code goes here]
    # ...
    
    # Here I assume we only stitch it back (no segmentation is done).
    segmented_image = list(map(lambda x: x.img, image_tiles))
    size = len(segmented_image) + 1
    subplot_dim = int(math.ceil(math.sqrt(size)))

    for idx in range(1, size):
        plt.subplot(subplot_dim, subplot_dim, idx)
        plt.imshow(segmented_image[idx - 1], cmap='gray')
        plt.axis('off')

    plt.suptitle('Tiles')
    plt.show()

    segmented_image = list(map(lambda tile_seg: ImageSlice(x=tile_seg[0].x,
                                                           y=tile_seg[0].y,
                                                           img=tile_seg[1]),
                               zip(image_tiles, segmented_image)))

    stitched_image = slicer.stitch_images(segmented_image)

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(stitched_image, cmap='gray')
    plt.title('Recovered image')
    plt.show()


if __name__ == '__main__':
    main()
