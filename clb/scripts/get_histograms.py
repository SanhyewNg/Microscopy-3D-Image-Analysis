import os

import fire
import matplotlib.pyplot as plt
import numpy as np

import clb.dataprep.readers as readers


def get_histogram(input_dir, frequency=10, channel=0):
    """A short script that generates histograms of slices from image stacks in
    the folder. Slices are chosen with the specified frequency.

    Args:
        input_dir (str): Path to the root folder.
        frequency (int): Frequency with which slices are chosen.
        channel (int): Number of the channel user wants histograms from.

    Returns:
        Saves histograms created with pyplot in the .png format.
    """
    path_list = os.listdir(input_dir)
    for input_filename in path_list:
        output_folder, _ = os.path.splitext(input_filename)
        output_dir = os.path.join(input_dir, 'histograms')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        output_file_dir = os.path.join(output_dir, output_folder)
        if not os.path.isdir(output_file_dir):
            os.mkdir(output_file_dir)
        with readers.get_volume_reader(os.path.join(input_dir,
                                                    input_filename)) as reader:
            stack_frequency = reader.shape.z // frequency
            image_stack = reader[0::stack_frequency, channel]
            for img, image_number in zip(image_stack, image_stack.z_indices):
                plt.hist(img,
                         bins=range(0, 256, np.iinfo(img.dtype).max // 255))
                output_img_dir = (output_dir
                                  + "histogram_slice_"
                                  + str(image_number))
                plt.savefig(output_img_dir)
                plt.close()


if __name__ is '__main__':
    fire.Fire(get_histogram)
