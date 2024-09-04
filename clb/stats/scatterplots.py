import os

import fire
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import clb.dataprep.readers as readers


FEATURES_PAIRS = (
    ['sphericity', 'equivalent_diameter [um]'],
    ['prolateness', 'oblateness'],
    ['{marker}_mean', 'oblateness'],
    ['{marker}_mean', 'prolateness'],
    ['{marker}_std', 'oblateness'],
    ['{marker}_std', 'prolateness'],
    ['volume [um^3]', '{marker}_perc_80'],
    ['sphericity', '{marker}_mean'],
    ['position_z [um]', 'volume [um^3]']
)


def plot_scatterplots(input, stats_path, output_dir, channel_names, series=0,
                      features_pairs=FEATURES_PAIRS):
    if isinstance(channel_names, str):
        channel_names = [channel_names]

    full_features_pairs = []
    for first_feature, second_feature in features_pairs:
        if '{marker}' in first_feature or '{marker}' in second_feature:
            for marker in channel_names:
                full_features_pairs.append(
                    (first_feature.format(marker=marker), second_feature.format(marker=marker))
                )
        else:
            full_features_pairs.append((first_feature, second_feature))

    with readers.get_volume_reader(path=input, series=series) as volume_iter:
        name = volume_iter.metadata.get('Name', 'series_{}'.format(series))
        stats_df = pd.read_csv(stats_path.format(name=name))
        class_groups = stats_df.groupby('classes')
        os.makedirs(output_dir, exist_ok=True)
        for first_feature, second_feature in full_features_pairs:
            fig, ax = plt.subplots(figsize=(40, 30))
            ax.set_xlabel(first_feature, fontsize=50)
            ax.set_ylabel(second_feature, fontsize=50)
            ax.tick_params(labelsize=40)

            for name, group in class_groups:
                ax.plot(group.loc[:, first_feature], group.loc[:, second_feature],
                        marker='o', linestyle='', ms=5, label=name, alpha=0.4)
            ax.legend(prop={'size': 30}, markerscale=3)
            output_name = '{} vs {}.png'.format(first_feature, second_feature)
            plt.savefig(os.path.join(output_dir, output_name.replace('/', '|')))


if __name__ == '__main__':
    fire.Fire(plot_scatterplots)
