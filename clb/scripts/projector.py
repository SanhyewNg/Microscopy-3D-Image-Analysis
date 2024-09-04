# For propper usage in jupyter lab:
# jupyter labextension install @jupyter-widgets/jupyterlab-manager
# jupyter labextension install jupyter-matplotlib
# jupyter labextension install @jupyterlab/plotly-extension
# %matplotlib widget  - insted inline`
#
# Require packages:
# pip install MulticoreTSNE
# pip install plotly
# pip install ipympl


import numpy as np
import pandas as pd
import daiquiri
import logging
import imageio
import skimage.segmentation
import skimage.filters
import click
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
import plotly.express as px

from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE

py.offline.init_notebook_mode(connected=True)

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

SEED = 42


def calculate_pca(features, labels, n_components=3):
    """Calculate PCA
    
    Args:
        features (pd.DataFame): data with multiindex
        labels (pd.Series): labels/class with multiindex 
    """
    logger.info("Calculating PCA")
    pca = PCA(n_components=n_components, random_state=SEED)
    pca_output = pca.fit_transform(features)
    pca_frame = pd.DataFrame(pca_output, index=labels.index, columns=["a", "b", "c"])
    pca_frame = labels.to_frame().join(pca_frame)
    return pca_frame


def calculate_tsne(features, labels, n_components=3):
    """Calculate PCA
    
    Args:
        features (pd.DataFame): data with multiindex
        labels (pd.Series): labels/class with multiindex 
    """
    logger.info("Calculating t-SNE")
    tsne = TSNE(n_components=n_components, n_jobs=-1)
    tsne_output = tsne.fit_transform(features)
    tsne_frame = pd.DataFrame(tsne_output, index=labels.index, columns=["a", "b", "c"])
    tsne_frame = labels.to_frame().join(tsne_frame)
    return tsne_frame


def plot_scatter_3d(dataframe, max_samples=1000, shuffle_samples=True):
    """Plot scatter 3d. To improve rotations number of samples is constrained.
    
    Args:
        dataframe (pd.DataFrame): Require `class` [0,1] as first column and 3 other columns with data to plot 
    """
    frame = dataframe.copy()
    if shuffle_samples:
        frame = frame.sample(frac=1, random_state=SEED)
    sub_frame = frame[:max_samples]

    class_, a, b, c = sub_frame.T.values
    color_map = np.vectorize(lambda x: "r" if x == 0 else "b")
    colors = color_map(class_)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(a, b, c, c=colors, marker="o")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("c")

    plt.show()


def plot_scatter_3d_plotly(dataframe, max_samples=1000, shuffle_samples=True):
    """Plot scatter 3d. To improve rotations number of samples is constrained.
    
    Args:
        dataframe (pd.DataFrame): Require `class` [0,1] as first column and 3 other columns with data to plot 
    """
    frame = dataframe.copy()
    if shuffle_samples:
        frame = frame.sample(frac=1, random_state=SEED)
    sub_frame = frame[:max_samples]

    class_, a0, b0, c0 = sub_frame[sub_frame["class"] == 0].T.values
    _, a1, b1, c1 = sub_frame[sub_frame["class"] == 1].T.values
    class0 = go.Scatter3d(
        x=a0,
        y=b0,
        z=c0,
        mode="markers",
        marker=dict(size=2, line=dict(color="rgba(217, 217, 217, 0.14)", width=0.5), opacity=0.9),
    )

    class1 = go.Scatter3d(
        x=a1,
        y=b1,
        z=c1,
        mode="markers",
        marker=dict(size=2, line=dict(color="rgba(127, 127, 127, 0.14)", width=0.5), opacity=0.9),
    )

    fig = go.Figure([class0, class1])
    fig.update_layout(autosize=False, width=1200, height=800, margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4))
    py.offline.iplot(fig)


class TensorflowProjectorExporter:
    def __init__(self, export_path):
        """Enable export to tensorflow projector. Please download projector from 
        `https://github.com/tensorflow/embedding-projector-standalone` and run index.html.
        To load data click load on upper left side.
        In step 1 - Load a TSV file of vectors: Load feature TSV
        In step 2 - Load a TSV file of metadata: Load a TSV file with metadata

        Args:
            export_path (str): export path of TSV files. 
        """
        self.export_path = Path(export_path)

    def export(self, dataset, labels, filename_prefix=""):
        """Export metadata.tsv and features.tsv. Columns `level_0` and `level_1` are renamed
        to `image` and `patch`
        
        Args:
            dataset (pd.DataFrame): Features only
            labels (pd.DataFrame): Labels corespondent to features(dataset)
            filename_prefix (str): Prefix of output files
        """
        if filename_prefix:
            features_path = self.export_path / (filename_prefix + "features.tsv")
            metadata_path = self.export_path / (filename_prefix + "metadata.tsv")
        else:
            features_path = self.export_path / "features.tsv"
            metadata_path = self.export_path / "metadata.tsv"

        self._export_features(dataset, features_path)
        self._export_metadata(dataset, labels, metadata_path)

    def _export_features(self, dataset, path):
        logger.info("Exporting features: {}".format(path))
        dataset.to_csv(path, sep="\t", index=False, header=False)

    def _export_metadata(self, dataset, labels, path):
        logger.info("Exporting metadata: {}".format(path))
        concatenated = pd.concat([dataset, labels], axis=1, sort=False)
        m = concatenated.reset_index()

        # move and rename image and patch columns to end of dataframe
        cols = list(m.columns.values)
        cols.pop(cols.index("level_0"))
        cols.pop(cols.index("level_1"))
        metadata = m[cols + ["level_0", "level_1"]]
        metadata.rename(inplace=True, columns={"level_0": "image", "level_1": "patch"})

        metadata.to_csv(path, sep="\t", index=False)


class DataLoader:
    def __init__(self, dataset_path, class_label="class", filters=None):
        """Load data, apply filters on them.

        Data in tsv file should:
            have as first two columns index values
            one column with label(class)

        Args:
            dataset_path (str): path to tsv file with features
            class_label (str): column label name of class to which given row belongs
            filters (list[callable]): Optional - filter / preprocessing functions
        """
        logger.info("Loading dataset file: {}".format(dataset_path))
        self.features = []
        self.class_label = class_label
        self._dataset = pd.read_csv(dataset_path, sep="\t", index_col=[0, 1])
        if filters:
            logger.info("Applying filters...")
            for f in filters:
                self._dataset = f(self._dataset)

    def set_features(self, feature_list):
        features_ = set(feature_list)
        logger.info("Setting {} features".format(len(features_)))
        self.features = features_

    def set_features_from_file(self, file_path):
        """File schema: 
        feature_name
        feature_name2
        feature_name3
        """
        features = []
        with open(file_path, "r") as feature_file:
            for line in feature_file.readlines():
                features.append(line.split("\n")[0])
        self.set_features(features)

    def get_all_colum_names(self):
        return list(self._dataset.columns)

    @property
    def dataset(self):
        """"Returns dataset with rows specified as features."""
        logger.info("Fetching dataset...")
        return self._dataset[self.features]

    @property
    def labels(self):
        """Retruns class column."""
        logger.info("Fetching labels...")
        return self._dataset[self.class_label]


def remove_uncertain_class(dataset, class_label="class"):
    return dataset.drop(dataset[dataset[class_label] >= 2].index)


def export_to_projector(data, export_name_prefix):
    exporter = TensorflowProjectorExporter(Path("."))
    exporter.export(data.dataset, data.labels, export_name_prefix)


def load_data(data_path, features_file_path, filters=[remove_uncertain_class]):
    data = DataLoader(dataset_path=data_path, class_label="class", filters=filters)
    data.set_features_from_file(features_file_path)
    return data


def plot3d(data, plot_type="pca", samples=1000):
    if plot_type == "pca" or "both":
        pca = calculate_pca(data.dataset, data.labels)
        plot_scatter_3d(pca, samples)
    if plot_type == "tsne" or "both":
        tsne = calculate_tsne(data.dataset, data.labels)
        plot_scatter_3d(tsne, samples)


@click.command()
@click.option("--data", help="Path to CSV file with features and labels", required=True)
@click.option("--features", help="Path to file with features. Each feature_name in own line.", required=True)
@click.option("--export", help="Export files for tensorflow projector", is_flag=True)
@click.option("--plot", help="Plot scatter 3d", type=click.Choice(["pca", "tsne", "both"]))
@click.option("--plot_samples", default=1000)
def main(data, features, export, plot, plot_samples):
    data = load_data(data, features)
    if export:
        export_to_projector(data, "")
    if plot:
        plot3d(data, plot, plot_samples)


if __name__ == "__main__":
    main()
