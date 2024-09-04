"""Module defines tools for calculating spatial statistics."""
import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as sndimage
import scipy.ndimage.morphology as morphology
import scipy.signal as ssignal
import statsmodels.distributions.empirical_distribution as emp_dist

import clb.dataprep.readers as readers


def calculate_esd(class_mask, voxel_size=None):
    """Calculate empty space distance for class given its mask.

    Args:
        class_mask (array_like): Classification mask, negatives marked 2, positives 255.
        voxel_size (array_like): Voxel size, shape (Z, Y, X). If None it's assumed to be
                                 ones.

    Returns:
        np.ndarray: Calculated ESD, shape as `class_mask`.
    """
    background_mask = np.squeeze(class_mask != 255)

    if voxel_size is None:
        voxel_size = np.ones_like(background_mask.shape)

    distances = morphology.distance_transform_edt(input=background_mask,
                                                  sampling=voxel_size)

    return distances


def calculate_class_distances(labels, distances):
    """Calculate distances from cells of one class to closest cells of reference class.

    Args:
        labels (array_like): Volume with cell labels.
        distances (array_like): Volume with distances from the reference class to each
                                voxel, should be the same shape as `labels`.

    Returns:
        np.ndarray: Distances from each cell of a class to closest cell of the reference
                    class, shape (N,)
    """
    labels = np.squeeze(labels)
    distances = np.squeeze(distances)

    cell_distances = sndimage.minimum(input=distances, labels=labels,
                                      index=np.trim_zeros(np.unique(labels)))

    return cell_distances


def get_class_labels(labels, class_mask):
    """Extract labels of cells that are classified positive.

    Args:
        labels (array_like): Volume with cell labels.
        class_mask (array_like): Classification mask, negatives marked 2, positives 255.

    Returns:
        np.ndarray: Volume with same shape like `labels` but only with labels of cells
                    that were classified positive.
    """
    class_ids = np.where(class_mask == 255, labels, 0)

    return class_ids


def calculate_cdf(values):
    """Return CDF function for given `values`.

    Args:
        values (np.ndarray)

    Returns:
        emp_dist.ECDF
    """
    cdf = emp_dist.ECDF(values.flatten())

    return cdf


def downsample(distances, desired_length=10 ** 6):
    """Downsample distances array until it has at most `desired_length` elements.

    Distances will be downsampled 10 times until reaching desired number of elements.

    Args:
        distances (array_like): Distances array, can be of any shape.
        desired_length (int): Upper bound for number of elements.

    Returns:
        np.ndarray: Downsampled version, shape (N,).
    """
    distances = distances.flatten()
    while len(distances) > desired_length:
        distances = ssignal.decimate(x=np.sort(distances), q=10, ftype='fir')

    return distances


def remove_double_positives(class_mask, ref_class_mask):
    """Remove positives from `class_mask` that are also positives in `ref_class_mask`.

    In both arrays positives should be marked as 255.

    Args:
        class_mask (array_like): Classification mask.
        ref_class_mask (array_like): Classification mask of reference class.

    Returns:
        np.ndarray: `class_mask` with double positives removed, same shape as
                    `class_mask`.
    """
    filtered_mask = np.where(
        np.logical_and(class_mask == 255, ref_class_mask == 255),
        0, class_mask)

    return filtered_mask


def save_distances(output_dir, names_to_distances):
    """Save arrays with distances to .csv files.

    Args:
        output_dir (str): Directory to save files to.
        names_to_distances (dict): Mapping names of files to distances arrays.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, distances in names_to_distances.items():
        distances_df = pd.Series(distances.flatten(), name=name, copy=False)
        distances_df.to_csv(os.path.join(output_dir, name + '.csv'), header=False,
                            index=False)


def make_cdf_plots(names_to_cdfs, max_x):
    """Plot CDFs with fixed style.

    All curves will be on one graph.

    Args:
        names_to_cdfs (Iterable): Pairs of legend entry, cdf to plot.
        max_x (float): Maximum sampling value for CDF.
    """
    x = np.linspace(0, max_x, 500)
    legend_entries = []
    plt.xlim(left=0, right=max_x)
    plt.ylim(bottom=0, top=1.05)
    plt.xlabel('distance [um]')
    plt.ylabel('CDF')
    plt.hlines(1, 0, max_x, linestyles='dashed')
    for name, cdf in names_to_cdfs:
        plt.plot(x, cdf(x))
        legend_entries.append(name)

    plt.legend(legend_entries)


class TestedClassesError(Exception):
    """Raised when list with classes names has different length than list with paths."""


def main(input, output_graph, output_data_dir, labels, ref_plot_name,
         ref_class_path, tested_classes_names, tested_classes_paths, series=0,
         filter_double_positives=False, sort_legend=False):
    """Calculate spatial statistics.

    Args:
        input (str): Path to input file (.tif, .lif, .uff).
        output_graph (str): Path to save graph with CDFs.
        output_data_dir (str): Path to save .csv files with distances. Names of the files
                               will be the same as corresponding legend entries in graph.
        labels (str): Path to segmentation results.
        ref_plot_name (str): Legend entry for empty space distance CDF of the reference
                             class.
        ref_class_path (str): Path to classification results with reference class.
        tested_classes_names: Legend entries.
        tested_classes_paths: Paths to tested classes, matching `tested_classes_names`.
        series (int): Series number.
        filter_double_positives (bool): Should double positives be filtered from tested
                                        classes.
        sort_legend (bool): Should legend entries be sorted (it's mainly for easier
                            testing).
    """
    with readers.get_volume_reader(ref_class_path) as ref_class_mask, \
            readers.get_volume_reader(input, series=series) as volume:
        esd = calculate_esd(ref_class_mask, getattr(volume, 'voxel_size', None))

        if len(tested_classes_names) != len(tested_classes_paths):
            raise TestedClassesError("List of tested classes names has different length"
                                     "than list of tested classes paths")
        try:
            names_to_class_masks = [(name, readers.get_volume_reader(a_class))
                                    for name, a_class in zip(tested_classes_names,
                                                             tested_classes_paths)]
            if filter_double_positives:
                names_to_class_masks = ((name, remove_double_positives(class_mask,
                                                                       ref_class_mask))
                                        for name, class_mask in names_to_class_masks)

            with readers.get_volume_reader(labels) as labels:
                names_to_class_labels = ((name, get_class_labels(labels, class_mask))
                                         for name, class_mask in names_to_class_masks)
                names_to_distances = {name: calculate_class_distances(class_ids,
                                                                      distances=esd)
                                      for name, class_ids in names_to_class_labels}
                downsampled_esd = downsample(esd)
                names_to_distances[ref_plot_name] = downsampled_esd[downsampled_esd > 0]
                save_distances(output_data_dir, names_to_distances)

                if sort_legend:
                    names_to_distances = sorted(names_to_distances.items())
                else:
                    names_to_distances = names_to_distances.items()

                names_to_cdfs = ((name, calculate_cdf(distances))
                                 for name, distances in names_to_distances)

                max_distance = downsampled_esd.max()
                make_cdf_plots(names_to_cdfs, max_x=max_distance)
                plt.savefig(output_graph)
        finally:
            for _, reader in names_to_class_masks:
                reader.close()


if __name__ == '__main__':
    fire.Fire(main)
