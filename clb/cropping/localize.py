import argparse

import imageio
import numpy as np
from tqdm import tqdm

from clb.cropping import CropInfo
from clb.image_processing import correlate2d_with_fft
from clb.utils import normalize_channels_volume, crop_image


def find_position_2d(image, crop):
    """
    Determine the position of the crop in the image. 
    Args:
        image: NxM numpy array 
        crop: PxR numpy array 

    Returns:
        absolute difference using the best match, top-left coordinates of that match
            or
        None if no suitable match has been found
    """
    assert image.ndim == 2 and crop.ndim == 2, "input args have to be 2D"

    # uses correlation to generate image where intensity related to the similarity
    correlation = correlate2d_with_fft(image, crop - crop.mean())
    center_yx = np.unravel_index(np.argmax(correlation), correlation.shape)

    # correlate2d_with_fft returns max at the center of the match, we need to translate for top-left
    top_left = center_yx[0] - (crop.shape[0] // 2), center_yx[1] - (crop.shape[1] // 2)

    # re-crop and calculate similarity
    full_cropped = crop_image(image, crop.shape, top_left, multi_channel=False)
    if full_cropped.shape != crop.shape:
        return None

    absdiff_mean = np.absolute(full_cropped - crop.astype(float)).mean()

    return absdiff_mean, top_left


def find_position_3d(volume, crop):
    """
    Determine the position of the crop in 3D volume. 
    Args:
        volume: HxNxM numpy array 
        crop: PxR numpy array

    Returns:
        absolute difference using the best match
        CropInfo representing the best match of the crop in volume.
    """
    assert volume.ndim == 3 and crop.ndim == 2, "volume has to be 3D and crop have to be 2D"

    per_image_best = []
    # Calculate the best match in each layer of volume, choose the match with the lowest diff.
    for idx, im in enumerate(list(volume)):
        pos2d = find_position_2d(im, crop)
        if pos2d is None:
            continue

        diff, top_left = pos2d
        crop_info = CropInfo(top_left[0], top_left[1], idx, crop.shape)
        per_image_best.append((diff, crop_info))

        # If exact match found, stop the search.
        if np.isclose(diff, 0):
            break

    best = sorted(per_image_best, key=lambda x: x[0])[0]
    return best


def find_positions_3d(volume, crop_volume):
    """
    Determine the position of each image from crop_volume in 3D volume. 
    Args:
        volume: HxNxM numpy array 
        crop_volume: ExPxR numpy array

    Returns:
        sum of absolute difference of all matches
        list of CropInfo representing the best match for each layer of crop_volume in volume.
    """
    assert volume.ndim == 3 and crop_volume.ndim == 3, "input args have to be 3D"

    # Find first layer of crop in entire volume.
    first_diff, first_crop_info = find_position_3d(volume, crop_volume[0])
    volume_cropped = crop_image(volume, first_crop_info.shape, (first_crop_info.y, first_crop_info.x),
                                multi_channel=False)

    res = [first_crop_info]
    diff_sum = first_diff

    # For the rest of layers of crop search only in Zs (as x, y are already fixed)
    for c in tqdm(crop_volume[1:]):
        diff, crop_info = find_position_3d(volume_cropped, c)
        # crop_info is found in cropped volume so we need to copy the coordinates from the first
        crop_info.y, crop_info.x = first_crop_info.y, first_crop_info.x
        diff_sum += diff
        res.append(crop_info)

    # Zs in CropInfo should form an increasing arithmetic sequence.
    if len(res) > 1:
        change = res[1].z - res[0].z
        expected = list(range(res[0].z, res[0].z + change * (len(res) - 1) + 1, change))
        actual = [crop.z for crop in res]
        assert expected == actual, "Zs in CropInfo should form an increasing arithmetic sequence."

    return diff_sum, res


def parse_arguments():
    parser = argparse.ArgumentParser(description='CLB find and save crops position in the original imagery.',
                                     add_help=False)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--original_volume', help='original volume path', required=True)
    required.add_argument('--crop_volume', help='crops for which positions are to be found', required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--output_info', help='where to save crop position')
    return parser.parse_args()


def main(args):
    volume = imageio.volread(args.original_volume)
    one_channel_volume = normalize_channels_volume(volume)

    crop_volume = imageio.volread(args.crop_volume)
    one_channel_crop_volume = normalize_channels_volume(crop_volume)

    diff, positions = find_positions_3d(one_channel_volume, one_channel_crop_volume)

    if args.output_info is None:
        for pos in positions:
            print(pos)
    else:
        CropInfo.save(positions, args.output_info)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
