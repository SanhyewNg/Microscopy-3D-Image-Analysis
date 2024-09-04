"""Module defines tools for calculating statistics of entire volume of cells.

Statistics include for example number of cells in classes, density, etc.
"""
import fire
import numpy as np
import pandas as pd

import clb.dataprep.readers as readers
import clb.dataprep.utils as utils
import clb.stats.utils as stutils


CELLS_NUM = 'number_of_cells'
DENSITY = 'density [cells/um^3]'
RATIO_OF_VOLUME = 'ratio_of_volume_occupied'
RATIO = 'ratio'
TOTAL_VOLUME = 'volume [um^3]'
SAMPLE_VOLUME = 'total_sample_volume [um^3]'


def get_num_of_cell_pixels(labels, ids=None):
    """Calculate number of pixels occupied by given `ids` in the volume.

    Args:
        labels (np.ndarray): Volume with cell labels.
        ids (set|None): Ids of cells to count volume. If None volume of all
                        will be counted.

    Returns:
        int: Volume occupied by cells.
    """
    if ids is None:
        cells_volume = np.count_nonzero(labels)
    else:
        ids = list(ids)
        voxels_with_ids = np.isin(element=labels, test_elements=ids)
        cells_volume = np.count_nonzero(voxels_with_ids)

    return cells_volume


def calc_volume(num_of_pixels, metadata):
    """Calculate volume from `number_of_pixels` and pixel size.

    Args:
        num_of_pixels (int): Number of pixels.
        metadata (dict): It is expected to have keys 'PhysicalSizeX',
                         'PhysicalSizeY', 'PhysicalSizeZ'.

    Returns:
        float: Calculated volume.
    """
    try:
        volume = (num_of_pixels
                  * float(metadata['PhysicalSizeX'])
                  * float(metadata['PhysicalSizeY'])
                  * float(metadata['PhysicalSizeZ']))

        return volume
    except KeyError:
        raise MetadataError('Missing metadata necessary to calculate volume.')


def calc_stats_of_group(labels, metadata, ids, all_ids):
    """Calculate statistics of one group of cells.

    For now statistics include (names given here are dictionary keys):
    - number_of_cells
    - density (cells/um^3)
    - ratio_of_volume_occupied
    - ratio
    - total_volume
    - total_sample_volume

    Args:
        labels (np.ndarray): Volume with cell labels.
        metadata (dict): Metadata of input data.
        ids (set): Ids of cells to calculate statistics.
        all_ids (set): Ids of all cells in `labels` volume.

    Returns:
        dict: Calculated statistics, keys described above.
    """
    cells_num = len(ids)
    total_sample_volume = calc_volume(labels.size, metadata)
    num_of_cell_pixels = get_num_of_cell_pixels(labels, ids)
    total_volume = calc_volume(num_of_cell_pixels, metadata)

    stats = {
        CELLS_NUM: cells_num,
        DENSITY: cells_num / total_sample_volume,
        RATIO_OF_VOLUME: total_volume / total_sample_volume,
        RATIO: cells_num / len(all_ids),
        TOTAL_VOLUME: total_volume,
        SAMPLE_VOLUME: total_sample_volume
    }

    return stats


class MetadataError(Exception):
    """Raised when necessary metadata is missing."""


def calculate_basic_stats(labels, metadata, **classes):
    """Calculate basic statistics.

    Args:
        labels (np.ndarray): Volume with labels.
        metadata (dict): Metadata of input images.
        classes: Volumes with classification results.

    Returns:
        pd.DataFrame: Calculated statistics.

    Raises:
        MetadataError: If necessary metadata is missing.
    """
    all_ids = stutils.get_all_ids(labels)
    classes_to_ids = stutils.map_classes_to_ids(labels, **classes)
    class_groups_to_ids = stutils.map_class_groups_to_ids(all_ids,
                                                          classes_to_ids)
    stats = {comb_name: calc_stats_of_group(labels, metadata, ids, all_ids)
             for comb_name, ids in class_groups_to_ids.items()}

    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.index.name = 'classes'
    return stats_df


def main(input, labels, output, series=None, **classes):
    """Calculate basic statistics.

    Args:
        input (str): Path to input .lif file.
        labels (str): Path to .tif file with labels, {} inside path will be
                      replaced with series number, if it has more pairs of
                      curly braces they should have 0 inside - {0}.
        output (str): Path to save statistics, should be .csv file. Curly
                      braces work like in `labels` argument.
        series (int): Series for which statistics should be calculated.
        classes: Paths to .tif files with classification results. Curly
                 braces work like in previous arguments.
    """
    with readers.get_volume_reader(path=input, series=series) as vol_iter:
        name = vol_iter.metadata.get('Name', 'series_{}'.format(series))

        try:
            classes_to_volumes = {
                class_name: readers.get_volume_reader(path=path.format(name=name))
                for class_name, path in classes.items()
            }

            labels_path = labels.format(name=name)
            with readers.get_volume_reader(path=labels_path) as reader:
                labels_volume = np.squeeze(reader)

            stats = calculate_basic_stats(labels_volume, vol_iter.metadata,
                                          **classes_to_volumes)

            output_path = output.format(name=name)
            utils.ensure_path(output_path, extensions='.csv')
            stats.to_csv(output_path,
                         columns=[DENSITY,
                                  CELLS_NUM,
                                  RATIO,
                                  RATIO_OF_VOLUME,
                                  TOTAL_VOLUME,
                                  SAMPLE_VOLUME])
        finally:
            for reader in classes_to_volumes.values():
                reader.close()


if __name__ == '__main__':
    fire.Fire(main)
