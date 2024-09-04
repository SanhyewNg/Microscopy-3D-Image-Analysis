import argparse

import imageio
import numpy as np

from clb.dataprep.utils import load_tiff_stack


class VolumeSlicer:
    def __init__(self, max_volume_height):
        self.max_volume_height = max_volume_height
        self.input_shape = ()

    def divide_volume(self, volume):
        """
        Split volume into a number of max_volume_height thick slices.
        Subsequent slices overlap on each other - one common image. 
        Args:
            volume: S x Y x X numpy array
        Returns:
            list of thick slices
        """
        self.input_shape = volume.shape
        volume_height = self.input_shape[0]

        res = []
        current = 0

        while current == 0 or current < volume_height - 1:
            if self.max_volume_height == 1:
                res = list(volume)
                break
            start_z = current
            end_z = min(start_z + self.max_volume_height, volume_height)
            res.append(volume[start_z:end_z])
            current += self.max_volume_height - 1

        return res

    def stitch_volume(self, volumes):
        """
        Merge thick slices into one volume. 
        Assumes that there is one image overlap between subsequent slices.
        Args:
            volumes: list of Si x Y x X numpy arrays with one image overlap

        Returns:
            one volume numpy array
        """
        res = np.zeros(self.input_shape, dtype=volumes[0].dtype)

        current = 0
        for volume in volumes:
            size = volume.shape[0]
            res[current:current + size] = volume
            current += size - 1

        return res


def split_volume(img_stack, gt_stack, num_channels, spatial_context=False):
    """
    Takes img_stack volumes of shape (num_pages, height, width), and makes
    n-channel images, in a form of numpy array (height, width, channels),
    while simultaneously unstacking and saving ground truths only for the middle
    channel.

    Args:
        img_stack: A volume of shape (num_pages, height, width),
                containing subsequent images from the original image stack.
        gt_stack:  A volume of shape (num_pages, height, width).
                containing ground truths
        num_channels: Number of channels in the output
        spatial_context: if True: saving ground truths only for the middle
            channel in the img stack.
    Returns:
        img_stack_channeled: If num_channels > 1: A list of multi-channel
        images of shape (height, width, channels).
                If == 1: A list of images of shape (height, width, channels).
        gt_stack_channeled: A list of ground truths of shape (height, width).
            if spatial_context=False and num_channels >1, it's a list of
            stack of ground truths (height, width, channels)
    """

    assert (num_channels <= img_stack.shape[0]), ("More channels than "
                                                  "pages in the input file.")
    img_stack_channeled = []
    gt_stack_channeled = []

    if spatial_context:
        assert (num_channels % 2 != 0), (
            "Number of channels in a slice is even, "
            "there must be a middle channel")
        side_channels = num_channels // 2
        for i in range(side_channels, img_stack.shape[0]-side_channels):
            single_image = img_stack[i-side_channels:(i+1)+side_channels, :, :]
            gt_stack_channeled.append(gt_stack[i, :, :])

            if single_image.shape[0] == 1:
                single_image = np.squeeze(single_image)
            else:
                # np.transpose is making single_image channel-last
                single_image = np.transpose(single_image, (1, 2, 0))
            img_stack_channeled.append(single_image)

    else:
        vol_slicer = VolumeSlicer(num_channels)
        img_stack_channeled = vol_slicer.divide_volume(img_stack)
        gt_stack_channeled = vol_slicer.divide_volume(gt_stack)

    return img_stack_channeled, gt_stack_channeled


def parse_arguments():
    parser = argparse.ArgumentParser(description='Volume slicer.',
                                     add_help=False)

    required = parser.add_argument_group('required arguments')
    required.add_argument('--input', help='3d input TIFF path', required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()

    slicer = VolumeSlicer(5)
    volume = load_tiff_stack(args.input)

    volume_slices = slicer.divide_volume(volume)

    for i, single_slice in enumerate(volume_slices):
        imageio.volwrite('tile' + str(i) + '.tif', single_slice)


if __name__ == '__main__':
    main()
