import os
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from vis.visualization import visualize_activation

from clb.dataprep.generators import single_image_generator
from clb.dataprep.utils import (get_number_of_pages,
                                 get_tiff_paths_from_directories)


def visualize_learned_patters(model, name_pattern, filters=None,
                              output_dir=None):
    """Visualize patterns learned by the network.

    The aim of this function is to allow to have an insight what specific
    filters in specific layers have learned. Such information can be used in
    transfer learning to decide which layers are suitable to be kept and
    which should be trained from scratch.

    Args:
        model: Keras model that will be visualized.
        name_pattern (str): provide substring that should be contained in
                            layer's name that will be visualized. Sample is
                            'conv'.
        filters (int): which filters should be visualized. If `None`, then all
                       filters of specific layer will be visualized.
        output_dir (str): output directory to save the results. If `None`,
                           results won't be saved.
    """
    for layer_idx, layer in tqdm(enumerate(model.layers), desc='Layers'):

        if name_pattern in layer.name:

            if not filters:
                filters = range(layer.output_shape[-1])

            for filter_idx in tqdm(filters, desc='Filters'):
                learned_pattern = visualize_activation(model=model,
                                                       layer_idx=layer_idx,
                                                       filter_indices=filter_idx)

                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    file_path = os.path.join(output_dir,
                                             'vis-layer-{}-filter-{}.png'
                                             .format(layer.name, filter_idx))

                    # `visualize_activation` returns single channel as the
                    # output. To write it as the image, it needs to be
                    # extracted into 2D array.
                    learned_pattern = learned_pattern[:, :, 0]

                    plt.imsave(file_path, learned_pattern, cmap='gray')


def count_steps(image_data, channels):
    """Count number of steps for a network to work through in given TIFF files
    from `image_data` dir or dirs and return number of input images created.

    Args:
        image_data (path): directory with TIFF files or a list of directories
                    with TIFF files.
        channels (int): Number of channels created for the spatial context.

    Returns:
        Overall number of pages from all TIFF files.
    """
    files = get_tiff_paths_from_directories(image_data)
    total_pages = 0
    for path in files:
        num_of_pages = get_number_of_pages(path)
        assert num_of_pages >= channels, ("Can't have more channels than pages",
                                          " in a TIFF file")
        total_pages += num_of_pages - (channels - 1)
    return total_pages


def get_images_for_tensorboard(batch_gen, num_imgs, architecture):
    """Prepare samples from `batch_gen` to be previewed live during training.

    Args:
        batch_gen: generator that is expected to yield tuples
                  (img_batch, ground truth_batch)
        num_imgs: size of the subset that will be taken from `batch_gen`
        architecture (str): what network architecture will be trained. For
                            now it's either 'dcan' or 'unet'.

    Returns:
        tuple of numpy arrays (num_imgs * img, num_imgs * ground truth)
    """
    if architecture == 'dcan':
        single_image_gen = single_image_generator(batch_gen,
                                                  multiple_gt_outputs=True)

        data = [(img[0], obj_gt[0], bnd_gt[0]) for img, [obj_gt, bnd_gt] in
                islice(single_image_gen, num_imgs)]

        imgs, obj_gt, bnd_gt = zip(*data)

        return np.array(imgs), [np.array(obj_gt), np.array(bnd_gt)]

    elif architecture == 'unet':
        single_image_gen = single_image_generator(batch_gen,
                                                  multiple_gt_outputs=False)

        data = [(img, gt) for img, gt in islice(single_image_gen, num_imgs)]
        imgs, gts = zip(*data)
        return np.array(imgs), np.array(gts)


def plot_pred_results(architecture, img, gt, objects, boundaries):
    """Plot the results of the current prediction

    Args:
        architecture: Specify used architecture. dcan and unet for now
        img: Depending on the architecture 3D (1, height, width) or
                4D (1, height, width, channel) input tensor.
        gt: Depending on the architecture 3D (1, height, width) or
                4D (1, height, width, channel) ground_truth tensor.
        objects: 3D (1, height, width) tensor, depicting objects
                probability map.
        boundaries:  3D (1, height, width) tensor, depicting boundaries
                probability map.

    Returns:
         Displays PyPlot figures.
    """
    if architecture == 'unet':
        img = img[:, :, :, 1]
        gt = [gt[:, :, :, i] for i in [1, 2]]

    plt.subplot(2, 3, 1)
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.axis('off')
    plt.title('Input image')

    plt.subplot(2, 3, 2)
    plt.imshow(np.squeeze(gt)[0], cmap='gray')
    plt.axis('off')
    plt.title('Ground-truth objects')

    plt.subplot(2, 3, 3)
    plt.imshow(np.squeeze(gt)[1], cmap='gray')
    plt.title('Ground-truth boundaries')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(np.squeeze(objects), cmap='gray')
    plt.title('Objects probabilities')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(np.squeeze(boundaries), cmap='gray')
    plt.title('Boundaries probabilities')
    plt.axis('off')

    plt.show()
