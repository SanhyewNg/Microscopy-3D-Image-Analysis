"""Module defines tools for calculating intensity features for each cell.

Currently only mean, std and percentiles are calculated.
"""
import collections as colls

import fire
import numpy as np
import pandas as pd
import skimage
import skimage.measure as sm

import clb.dataprep.readers as readers
import clb.dataprep.utils as utils
import clb.stats.utils as stutils


def calc_stats(intensities):
    """Calculate intensity statistics.

    Args:
        intensities (np.ndarray): Cell intensities.

    Returns:
        dict: Calculated statistics. Now keys in dictionary are:
              - mean
              - std
              - perc_n for n in {10, 20, ..., 90}
    """
    stats = {
        'mean': np.mean(intensities),
        'std': np.std(intensities)
    }

    ranks = np.arange(10, 100, 10)
    percentiles = np.percentile(intensities, q=ranks)
    stats.update({
        'perc_{}'.format(rank): perc
        for rank, perc in zip(ranks, percentiles)
    })

    return stats


def calc_stats_for_all_cells(labels, intensities):
    """Calculate intensity statistics for all cells in `labels` volume.

    Args:
        labels (np.ndarray): Volume with cell labels.
        intensities (np.ndarray): Volume with cell intensities.

    Returns:
        pd.DataFrame: Intensity statistics. Row labels are cell ids, column
                      labels are keys from dict returned by `calc_stats`.
    """
    props = sm.regionprops(label_image=labels, intensity_image=intensities)
    ids_to_stats = {
        prop.label: calc_stats(prop.intensity_image[prop.image])
        for prop in props
    }
    stats = pd.DataFrame(ids_to_stats).transpose()
    columns = ['mean', 'std']
    columns.extend('perc_{}'.format(x) for x in range(10, 100, 10))
    stats = stats.reindex(columns, axis=1)

    return stats


def calc_stats_for_all_markers(labels, markers_to_intensities):
    """Calculate intensity statistics for all markers.

    Args:
        labels (np.ndarray): Volume with cell labels.
        markers_to_intensities (OrderedDict): Mapping marker names to
                                              VolumeIters with intensities.

    Returns:
        pd.DataFrame: Calculated statistics. Group of statistics for each
                      marker is prefixed by <marker_name>_.
    """
    scaled_intensities = (skimage.img_as_float(np.squeeze(intensity.to_numpy()))
                          for intensity in markers_to_intensities.values())
    stats = (calc_stats_for_all_cells(labels, intensity)
             for intensity in scaled_intensities)
    prefixed_stats = [stats_df.add_prefix(name + '_')
                      for name, stats_df in zip(markers_to_intensities, stats)]
    all_markers_stats = pd.concat(prefixed_stats, axis=1)
    all_markers_stats.index.name = 'cell_id'

    return all_markers_stats


class ChannelNamesLengthError(Exception):
    """Raised when channels list has different lengths than names list."""


def main(input, labels, output, channels, series=0, channel_names=None,
         start=0, stop=None, **classes):
    """Calculate intensity statistics.

    Args:
        input (str): Path to .lif input file.
        labels (str): Path to .tif file with labels.
        output (str): Output path, each pair of curly braces {} will be changed
                      to series number.
        series (int): Series to calculate statistics.
        channels (list|int): Numbers of channels to calculate statistics from.
        channel_names (list|str|None): Names of channels from `channels` argument.
                                       They will be used as prefix in statistics
                                       names, for example dapi_mean or ki67_std.
                                       If not given channel numbers will be used
                                       instead. Length of this list should be
                                       same as `channels`.
        start (int): Starting slice.
        stop (int|None): Slice after the last one. If None all slices until the
                         end will be used.
        classes: Volumes with classification results.
    """
    if isinstance(channels, int):
        channels = [channels]
    if isinstance(channel_names, str):
        channel_names = [channel_names]

    with readers.get_volume_reader(path=input, series=series) as volume_iter:
        name = volume_iter.metadata.get('Name', 'series_{}'.format(series))

        try:
            classes_to_volumes = {
                class_name: readers.get_volume_reader(path=path.format(name=name))
                for class_name, path in classes.items()
            }

            labels_path = labels.format(name=name)
            with readers.get_volume_reader(path=labels_path) as reader:
                labels_volume = np.squeeze(reader)

            if channel_names is not None:
                if len(channels) != len(channel_names):
                    msg = 'Channels list has different lengths than names list ' \
                          '{} != {}'.format(len(channels), len(channel_names))
                    raise ChannelNamesLengthError(msg)
            else:
                channel_names = map(str, channels)

            output_path = output.format(name=name)
            utils.ensure_path(output_path, extensions='.csv')

            intensities = (volume_iter[start:stop, channel] for channel in channels)
            markers_to_intensities = colls.OrderedDict(zip(channel_names,
                                                           intensities))
            all_markers_stats = calc_stats_for_all_markers(labels_volume,
                                                           markers_to_intensities)

            stutils.insert_classes(labels_volume, all_markers_stats,
                                   **classes_to_volumes)

            all_markers_stats.to_csv(output_path)
        finally:
            for reader in classes_to_volumes.values():
                reader.close()


if __name__ == '__main__':
    fire.Fire(main)
