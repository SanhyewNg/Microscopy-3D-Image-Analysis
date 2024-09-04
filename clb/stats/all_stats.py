"""Module defines helper script that allows for calculating all statistics at once."""
import fire
import pandas as pd

import clb.dataprep.readers as readers
import clb.stats.intensity_stats as intensity_stats
import clb.stats.morphology_stats as morphology_stats
import clb.stats.volume_stats as volume_stats


def main(input, labels, output, channels, channel_names, series=0,
         merge=True, append_aggregations=True, start=0, stop=None, **classes):
    """Calculate volume, morphology and intensity statistics.

    Args:
        input (str): Path to lif file.
        labels (str): Path to volume with labels. Each {name} will be replaced
                      with name of series from .lif file.
        output (str): Prefix of the output filenames. Can also contain {name}
                      placeholder. If this argument is 'prefix', three files
                      will be created: 'prefixvolume_stats.csv',
                      'prefixmorphology_stats.csv', 'prefixintensity_stats.csv'.
        channels (list|int): Channels to calculate intensity statistics from.
        channel_names (list|str): Channel names for intensity_statistics.
        series (int): Series to calculate statistics from.
        append_aggregations (bool): Should aggregations be appended to volume statistics.
        merge (bool): Should morphology statistics and intensity statistics be
                      merged.
        start (int): First slice used for calculations.
        stop (int|None): First slice not used for calculations.
        classes: Volumes with classification results.
    """
    volume_stats_output = output + 'volume_stats.csv'
    volume_stats.main(input=input, labels=labels, output=volume_stats_output,
                      series=series, **classes)

    morphology_stats_output = output + 'morphology_stats.csv'
    if append_aggregations:
        volume_stats_path = volume_stats_output
    else:
        volume_stats_path = None
    morphology_stats.main(input=input, labels=labels,
                          output=morphology_stats_output, series=series,
                          volume_stats=volume_stats_path, **classes)
    intensity_stats_output = output + 'intensity_stats.csv'
    intensity_stats.main(input=input, labels=labels,
                         output=intensity_stats_output,
                         series=series, channels=channels,
                         channel_names=channel_names, start=start, stop=stop, **classes)

    if merge:
        with readers.get_volume_reader(path=input, series=series) as volume_iter:
            name = volume_iter.metadata.get('Name', 'series_{}'.format(series))
            morphology_stats_path = morphology_stats_output.format(name=name)
            morphology_stats_df = (pd.read_csv(morphology_stats_path).
                                   set_index('cell_id'))
            intensity_stats_path = intensity_stats_output.format(name=name)
            intensity_stats_df = (pd.read_csv(intensity_stats_path).
                                  set_index('cell_id'))
            cols_to_use = intensity_stats_df.columns.difference(
                morphology_stats_df.columns, sort=False)
            merged_df = morphology_stats_df.merge(
                right=intensity_stats_df[cols_to_use],
                left_index=True,
                right_index=True,
                how='inner',
                sort=False)
            nuclei_stats_output = output.format(name=name) + 'nuclei_stats.csv'
            merged_df.to_csv(nuclei_stats_output)


if __name__ == '__main__':
    fire.Fire(main)
