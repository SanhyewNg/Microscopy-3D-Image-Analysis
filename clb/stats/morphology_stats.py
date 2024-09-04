"""Module defines functions for calculating morphology statistics of cells.

Example statistics are volume or orientation of a cell.
"""
import SimpleITK as sitk
import fire
import numpy as np
import pandas as pd
import radiomics as rd
import radiomics.shape as rshape
import skimage.measure as sm

import clb.dataprep.readers as readers
import clb.dataprep.utils as utils
import clb.stats.utils as stutils

FEATURES_TO_AGGREGATE = ('sphericity', 'prolateness', 'oblateness')
AGGREGATION_TYPES = ('mean', 'std', 'median')

CLASSES = 'classes'
VOLUME = 'volume [um^3]'
DIAMETER = 'equivalent_diameter [um]'
SPHERICITY = 'sphericity'
OBLATENESS = 'oblateness'
PROLATENESS = 'prolateness'
POSITION_X = 'position_x [um]'
POSITION_Y = 'position_y [um]'
POSITION_Z = 'position_z [um]'
ORIENTATION_1_X = 'orientation_longest_semi_axis_x'
ORIENTATION_1_Y = 'orientation_longest_semi_axis_y'
ORIENTATION_1_Z = 'orientation_longest_semi_axis_z'
ORIENTATION_2_X = 'orientation_second_semi_axis_x'
ORIENTATION_2_Y = 'orientation_second_semi_axis_y'
ORIENTATION_2_Z = 'orientation_second_semi_axis_z'


class ExtendedRadiomicsShape(rshape.RadiomicsShape):
    """RadiomicsShape, that stores also eigenvectors and not only eigenvalues.
    """

    def _initSegmentBasedCalculation(self):
        # Here spacing is reversed, so it should be passed as (x, y, z).
        self.pixelSpacing = np.array(self.inputImage.GetSpacing()[::-1])

        # Pad inputMask to prevent index-out-of-range errors
        self.logger.debug('Padding the mask with 0s')

        cpif = sitk.ConstantPadImageFilter()

        padding = np.tile(1, 3)
        try:
            cpif.SetPadLowerBound(padding)
            cpif.SetPadUpperBound(padding)
        except TypeError:
            # newer versions of SITK/python want a tuple or list
            cpif.SetPadLowerBound(padding.tolist())
            cpif.SetPadUpperBound(padding.tolist())

        self.inputMask = cpif.Execute(self.inputMask)

        # Reassign self.maskArray using the now-padded self.inputMask
        self.maskArray = (
                sitk.GetArrayFromImage(self.inputMask) == self.label)
        self.labelledVoxelCoordinates = np.where(self.maskArray != 0)

        self.logger.debug(
            'Pre-calculate Volume, Surface Area and Eigenvalues')

        # Volume, Surface Area and eigenvalues are pre-calculated

        # Compute Surface Area and volume
        self.SurfaceArea, self.Volume, self.diameters = \
            rd.cShape.calculate_coefficients(self.maskArray, self.pixelSpacing)

        # Compute eigenvalues and -vectors
        Np = len(self.labelledVoxelCoordinates[0])
        coordinates = np.array(self.labelledVoxelCoordinates,
                               dtype='int').transpose()
        physicalCoordinates = coordinates * self.pixelSpacing[None, :]
        physicalCoordinates -= np.mean(physicalCoordinates, axis=0)
        physicalCoordinates /= np.sqrt(Np)
        covariance = np.dot(physicalCoordinates.T.copy(), physicalCoordinates)
        self.eigenValues, eigenVectors = np.linalg.eigh(covariance)
        self.eigenVectors = eigenVectors.T

        # Correct machine precision errors causing very small negative
        # eigen values in case of some 2D segmentations
        machine_errors = np.bitwise_and(self.eigenValues < 0,
                                        self.eigenValues > -1e-9)
        if np.sum(machine_errors) > 0:
            self.logger.warning(
                'Encountered %d eigenvalues < 0 and > -1e-9, rounding to 0',
                np.sum(machine_errors))
            self.eigenValues[machine_errors] = 0

        # Sort the eigenValues (and corresponding) eigenVectors from small to
        # large.
        sort_indices = np.argsort(self.eigenValues)
        self.eigenVectors = self.eigenVectors[sort_indices]
        self.eigenValues = self.eigenValues[sort_indices]

        self.logger.debug('Shape feature class initialized')


def calc_cell_volume(feature_extractor):
    """Calculate volume of a cell.

    Args:
        feature_extractor (ExtendedRadiomicsShape): Used to calculate volume.

    Returns:
        float: Cell volume.
    """
    return feature_extractor.getVoxelVolumeFeatureValue()


def calc_sphericity(feature_extractor):
    """Calculate sphericity of a cell.

    Args:
        feature_extractor (ExtendedRadiomicsShape): Used to calculate
        sphericity.

    Returns:
        float: Cell sphericity.
    """
    return feature_extractor.getSphericityFeatureValue()


def calc_equivalent_diameter(volume):
    """Calculate diameter of a sphere with `volume`.

    Args:
        volume (float): Volume of a sphere.

    Returns:
        float: Diameter of sphere.
    """
    return (0.75 * volume / np.pi) ** (1. / 3.)


class NegativeEigenvaluesError(Exception):
    """Raised when not all eigenvalues are positive."""


def calc_prolateness(eigenvalues):
    """Calculate cell prolateness.

    Args:
        eigenvalues (np.ndarray): Eigenvalues of cell region. They should be
                                  sorted ascending.

    Returns:
        float: Cell prolateness.
    """
    if (eigenvalues < 0).any():
        raise NegativeEigenvaluesError(
            'Some of eigenvalues are negative : {}'.format(eigenvalues))
    square_roots = np.sqrt(eigenvalues)

    return (square_roots[2] - square_roots[0]) / square_roots[2]


def calc_oblateness(eigenvalues):
    """Calculate cell oblateness.

    Args:
        eigenvalues (np.ndarray): Eigenvalues of cell region. They should be
                                  sorted ascending.

    Returns:
        float: Cell oblateness.
    """
    if (eigenvalues < 0).any():
        raise NegativeEigenvaluesError(
            'Some of eigenvalues are negative : {}'.format(eigenvalues))
    square_roots = np.sqrt(eigenvalues)

    return 2 * ((square_roots[1] - square_roots[0])
                / (square_roots[2] - square_roots[0])) - 1


def calc_stats(region_prop, sizes):
    """Calculate morphology statistics for cell with `cell_mask`

    Args:
        region_prop (RegionProperties): Properties of one cell.
        sizes (tuple): Pixel sizes in form of (z, y, x).

    Returns:
        dict: Calculated statistics.
    """
    cell_mask = region_prop.image.astype(np.uint16)
    image = sitk.GetImageFromArray(cell_mask)
    image.SetSpacing(spacing=tuple(reversed(sizes)))
    # inputImage should be intensities, but we don't have to use them here.
    feature_extractor = ExtendedRadiomicsShape(inputImage=image,
                                               inputMask=image)
    first_vector = feature_extractor.eigenVectors[2]
    second_vector = feature_extractor.eigenVectors[1]

    cell_volume = calc_cell_volume(feature_extractor)
    features = {
        VOLUME: cell_volume,
        DIAMETER: calc_equivalent_diameter(cell_volume),
        SPHERICITY: calc_sphericity(feature_extractor),
        OBLATENESS: calc_oblateness(feature_extractor.eigenValues),
        PROLATENESS: calc_prolateness(feature_extractor.eigenValues),
        POSITION_X: region_prop['centroid'][2] * sizes[2],
        POSITION_Y: region_prop['centroid'][1] * sizes[1],
        POSITION_Z: region_prop['centroid'][0] * sizes[0],
        ORIENTATION_1_X: first_vector[2],
        ORIENTATION_1_Y: first_vector[1],
        ORIENTATION_1_Z: first_vector[0],
        ORIENTATION_2_X: second_vector[2],
        ORIENTATION_2_Y: second_vector[1],
        ORIENTATION_2_Z: second_vector[0],
    }

    return features


def calc_stats_for_all_cells(labels, sizes):
    """Calculate morphology statistics for each cell in `labels`.

    Args:
        labels (np.ndarray): Volume with cell labels, shape (Z, Y, X).
        sizes (tuple): Pixel sizes in form of (z, y, x).

    Returns:
        pd.DataFrame: Statistics.
    """
    props = sm.regionprops(label_image=labels)
    ids_to_features = {
        prop.label: calc_stats(prop, sizes)
        for prop in props}

    features_df = pd.DataFrame.from_dict(ids_to_features, orient='index')
    features_df.index.name = 'cell_id'

    return features_df


class AggregationTypeError(Exception):
    """Raised when trying to use unrecognized aggregation type."""


def calc_aggregation(morphology_stats, aggregation_type, features=FEATURES_TO_AGGREGATE):
    if aggregation_type == 'mean':
        args = {}
    elif aggregation_type == 'std':
        # Set 0, so we don't get Nan when there is only one sample.
        args = {'ddof': 0}
    elif aggregation_type == 'median':
        args = {}
    else:
        raise AggregationTypeError('Unrecognized aggregation type: {}'.
                                   format(aggregation_type))

    features_to_aggregate = morphology_stats.reindex(columns=('classes',) + features)
    aggregation_method = getattr(features_to_aggregate, aggregation_type)
    aggregation_of_all = aggregation_method(**args).add_prefix(aggregation_type + '_')

    groups = features_to_aggregate.groupby('classes')
    aggregation_method = getattr(groups, aggregation_type)
    aggregations = aggregation_method(**args).add_prefix(aggregation_type + '_')
    aggregations.loc['all_cells'] = aggregation_of_all

    return aggregations


def add_aggregations(volume_stats, features_aggregations):
    """Add aggregations to `volume_stats`.

    Args:
        volume_stats (pd.DataFrame): Frame to add aggregations to.
        features_aggregations (pd.DataFrame): Aggregations.

    Returns:
        pd.DataFrame: Frame with appended aggregations.
    """
    stats_with_aggregations = volume_stats.merge(features_aggregations, left_index=True,
                                                 right_index=True, how='left')
    stats_with_aggregations.set_index(volume_stats.index, inplace=True)

    return stats_with_aggregations


def main(input, labels, output, series=0, volume_stats=None, **classes):
    """Calculate morphology statistics.

    Args:
        input (str): Path to .lif input file.
        labels (str): Path to .tif file with labels.
        output (str): Path to output file. Each pair of curly braces {} will
                      be replaced with series number.
        series (int): Series for which statistics should be calculated.
        volume_stats (str|None): Path to volume stats to save aggregations.
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
            sizes = vol_iter.voxel_size

            output_path = output.format(name=name)
            utils.ensure_path(output_path, extensions='.csv')
            stats = calc_stats_for_all_cells(labels_volume, sizes)

            stutils.insert_classes(labels_volume, stats, **classes_to_volumes)

            stats.to_csv(output_path,
                         columns=[CLASSES,
                                  DIAMETER,
                                  SPHERICITY,
                                  VOLUME,
                                  OBLATENESS,
                                  PROLATENESS,
                                  POSITION_X,
                                  POSITION_Y,
                                  POSITION_Z,
                                  ORIENTATION_1_X,
                                  ORIENTATION_1_Y,
                                  ORIENTATION_1_Z,
                                  ORIENTATION_2_X,
                                  ORIENTATION_2_Y,
                                  ORIENTATION_2_Z])

            if volume_stats is not None:
                volume_stats_output = volume_stats.format(name=name)
                volume_stats_df = pd.read_csv(volume_stats_output, index_col=0)
                for aggregation_type in AGGREGATION_TYPES:
                    aggregation = calc_aggregation(morphology_stats=stats,
                                                   aggregation_type=aggregation_type)
                    volume_stats_df = add_aggregations(volume_stats_df,
                                                       features_aggregations=aggregation)
                volume_stats_df.to_csv(volume_stats_output, index=True)
        finally:
            for reader in classes_to_volumes.values():
                reader.close()


if __name__ == '__main__':
    fire.Fire(main)
