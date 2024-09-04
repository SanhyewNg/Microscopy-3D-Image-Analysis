import argparse

import imageio
import numpy as np

from clb.dataprep.utils import reduce_to_max_channel
# from clb.cp import batch_cellprofiler as batch_cp
from clb.image_slicer import ImageSlice, ImageSlicer
from clb.predict.predict_tile import predict_dcan


def segmentation2d_dcan_tile(images, trim_method, postprocess, model_path):
    normalized_images = list(map(reduce_to_max_channel, images))
    predictions = predict_dcan(normalized_images,
                               model_path,
                               trim_method=trim_method, postprocess=postprocess)

    res = []
    for idx, prediction in enumerate(predictions):
        prediction = np.squeeze(prediction)
        res.append((prediction * 255).astype(np.uint8))
    return res


def segmentation2d_by_tiles(image, pad_size, segmentation_tile, tile_size):
    # Split image into a tiles of a given size and add padding.
    slicer = ImageSlicer(tile_size, tile_size, pad_size)
    tiles = slicer.divide_image(image)

    # Run segmentation_tile separately on all of those tiles.
    segmented = segmentation_tile(list(map(lambda x: x.img, tiles)))
    if len(segmented) != len(tiles):
        raise Exception("segmentation_tile returned more images that given")

    segmented_tiles = list(map(lambda ts: ImageSlice(x=ts[0].x, y=ts[0].y,
                                                     img=ts[1]),
                               zip(tiles, segmented)))

    stitched = slicer.stitch_images(segmented_tiles)
    return stitched


def parse_arguments():
    parser = argparse.ArgumentParser(description='Segmentation of 2D image.',
                                     add_help=False)

    required = parser.add_argument_group('required arguments')
    required.add_argument('--input', help='2d input TIFF path')
    required.add_argument('--output', help='output TIFF path',
                          default="output.tif")
    required.add_argument('--model', help='path to network model',
                          default="../model.h5")
    required.add_argument('--trim_method',
                          help=('method of adapting input size for network '
                                '(resize, padding, reflect)'),
                          default="padding")
    required.add_argument('--postprocess',
                          type=bool,
                          help='postprocess probability to 0-1 mask',
                          default=False)
    return parser.parse_args()


def main():
    args = parse_arguments()

    def segmentation2d_dcan(img):
        return segmentation2d_dcan_tile(img, args.trim_method,
                                        args.postprocess, args.model)

    image = imageio.imread(args.input)
    result = segmentation2d_by_tiles(image, 30, segmentation2d_dcan, 140)

    try:
        imageio.imsave(args.output, result)
    except ValueError:
        raise IOError("--output {} should be a path to TIFF "
                      "file.".format(args.output))


if __name__ == '__main__':
    main()
