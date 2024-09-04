import fire
import h5py
import os
import imageio
import numpy as np
from skimage import img_as_float32, img_as_uint
from tqdm import tqdm
from skimage.external.tifffile import TiffWriter
from glob import glob

from clb.image_processing import clahe
from clb.utils import chunks


class H5Extractor:
    def extract_big_data_from_h5(self, input_path, group_dir, pixel_size_x=0.5,
                                 pixel_size_y=0.5):
        with h5py.File(input_path, mode='r+') as h5_input:
            h5_group = h5_input[group_dir]
            with TiffWriter(os.path.splitext(input_path)[0] + "_" + group_dir +
                            ".tif", bigtiff=True, imagej=True) as tif_image:
                for img_slice in self._generate_slices(h5_group):
                    tif_image.save(img_slice, compress=5,
                                   resolution=(pixel_size_x, pixel_size_y, None))

    def _generate_slices(self, h5_group):
        num_of_slices = h5_group.shape[0]
        for img_block_num in range(16, num_of_slices, 16):
            img_block = h5_group[img_block_num - 16:img_block_num, :, :]
            for img_slice_num in range(16):
                print("Slice num: " + str(img_slice_num))
                yield img_block[img_slice_num, :, :]

    def extract_samples(self, input_path, group_dir, output_path, slices_points=None, slice_length=12):
        slices_points = slices_points or [0.1, 0.3, 0.5, 0.7, 0.9]
        with h5py.File(input_path, mode='r') as h5_input:
            h5_group = h5_input[group_dir]
            num_of_slices = h5_group.shape[0]

            try:
                with TiffWriter(output_path) as tif_image:
                    for s_fraction_start in tqdm(slices_points):
                        first_slice = int(num_of_slices * s_fraction_start)
                        img_block = h5_group[first_slice:first_slice + slice_length, :, :]
                        for img in img_block:
                            tif_image.save(img, compress=5)
            except OSError:
                print("... failed")
                os.remove(output_path)
                return False

        return True

    def extract_samples_from_all(self, input_pattern, group_dir, output_dir, slices_points=None, slice_length=12):
        os.makedirs(output_dir, exist_ok=True)
        for h5_file in tqdm(glob(input_pattern)):
            h5_filename = os.path.basename(h5_file)
            print("Extracting {0}...".format(h5_file))
            output_path = os.path.join(output_dir, h5_filename).replace(".h5", ".tif")
            if os.path.isfile(output_path):
                print("... skipping")
                continue

            ok = self.extract_samples(h5_file, group_dir, output_path, slices_points, slice_length)
            if not ok:
                print("retry...")
                self.extract_samples(h5_file, group_dir, output_path, slices_points, slice_length)

    def process_samples(self, input_pattern, output_dir, histogram=False, average=False):
        os.makedirs(output_dir, exist_ok=True)

        for img_file in tqdm(glob(input_pattern), "Processing samples"):
            img_filename = os.path.basename(img_file)
            output_path = os.path.join(output_dir, img_filename)
            if os.path.isfile(output_path):
                print("... skipping")
                continue

            volume = imageio.volread(img_file)

            if histogram:
                equalized_images = [clahe(img_as_float32(img), size=256) for img in volume]
                volume = np.array([img_as_uint(img) for img in equalized_images])

            if average:
                average_chunks = [sum(chunk / 3) for chunk in chunks(volume, 3)]
                volume = np.array(average_chunks).astype(np.uint16)

            with TiffWriter(output_path) as tif_image:
                for img in volume:
                    tif_image.save(img, compress=5)


if __name__ == "__main__":
    fire.Fire(H5Extractor)
