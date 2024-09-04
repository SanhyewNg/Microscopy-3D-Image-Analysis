import math
from collections import OrderedDict
from pathlib import Path

import mahotas
import numpy as np
from scipy.spatial.qhull import QhullError
from skimage import img_as_ubyte, measure
from tqdm import tqdm

from clb.classify.extractors import (DESIRED_VOXEL_SIZE, DESIRED_VOXEL_UM,
                                     preprocess_input_labels)
from clb.classify.utils import add_data_with_prefix
from clb.image_processing import resample
from clb.utils import replace_values
from clb.yaml_utils import load_yaml_args


HARALICK_NAMES = """AngularSecondMoment Contrast Correlation Variance
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()

DIRECTIONS_3D = 13
HARALICK_FEATURES = ["haralick_{:02d}_{}".format(direction, name) for direction in range(DIRECTIONS_3D) for name in
                     HARALICK_NAMES]
HARALICK_AGGR_FEATURES = ["haralick_{}_{}".format(type, name) for type in ['mean', 'peak2peak'] for name in
                          HARALICK_NAMES]
MOMENT_NORMALIZED_FEATURES = ["moment_normalized_{}_{}_{}".format(z, y, x)
                              for z in range(0, 4) for y in range(0, 4) for x in range(0, 4) if z + y + x >= 2]
SELECTED_26_FEATURES = load_yaml_args(str(Path(__file__).resolve().parent / "feature_mapping.yaml"))["selected_26"]


def include(feature_types, feature_type_component):
    if feature_types == 'all':
        return True
    else:
        components = feature_types.split('+')
        return feature_type_component in components


def get_feature_columns(all_columns, features_type, channels_preprocessing_list=None):
    """
    Filter the list of actual columns to the ones representing the features present in features_type. If list of
    channel-preprocessing is provided then additional filter is added to keep only features calculated on provided
    ones.
    Args:
        all_columns: list of column names
        features_type: str id of the feature set
        channels_preprocessing_list: list of strings with channels (potentially with preprocessing suffix)

    Returns:
        list of column names that are features created for features_type
    """

    column_names = []
    shape_features_names = []
    features_names = []

    if include(features_type, "default"):
        features_names = ["mean_intensity", "std_intensity",
                          "perc_25_intensity", "median_intensity", "perc_75_intensity"]

    if include(features_type, "perc_per_10"):
        features_names = ["mean_intensity", "std_intensity",
                          "perc_25_intensity", "median_intensity", "perc_75_intensity"]
        for i in range(1, 10):
            features_names += ["perc_" + str(i * 10) + "_intensity"]

    if include(features_type, "mini"):
        features_names = ["mean_intensity", "std_intensity", "perc_75_intensity"]

    if include(features_type, "complex"):
        shape_features_names = ["solidity", "first_major_ratio", "second_major_ratio"]
        features_names = ["mean_intensity", "std_intensity", "mad_intensity"]
        for i in range(1, 10):
            features_names += ["perc_" + str(i * 10) + "_intensity"]
        features_names += ['mass_displace_in_diameters', 'mass_displace_in_majors']
        features_names += MOMENT_NORMALIZED_FEATURES

    if include(features_type, "texture"):
        features_names += HARALICK_AGGR_FEATURES

    if include(features_type, "selected_26"):
        features_names += SELECTED_26_FEATURES

    if not features_names and not shape_features_names:
        raise Exception("Not recognised feature_type: " + features_type)

    features_columns = [col for col in all_columns if any(col.endswith(f) for f in features_names)]
    if channels_preprocessing_list is None:
        column_names += features_columns
    else:
        for preprocess in channels_preprocessing_list:
            preprocess_columns = [col for col in features_columns if col.startswith(preprocess + "_")]
            assert any(preprocess_columns), "No columns found for {0}.".format(preprocess)
            column_names += preprocess_columns

    shape_columns = [col for col in all_columns if col in shape_features_names]
    if shape_features_names:
        column_names += shape_columns

    return sorted(column_names)


def intensity_features_from_props(prop, features_type):
    """
    Calculate simple scale invariant features for the given object
    Args:
        prop: property object of the cell
        features_type: name of the features to calculate
    Returns:
        dictionary of features with values for this cell
    """
    res = OrderedDict()

    res["mean_intensity"] = prop.mean_intensity
    intensities = prop.intensity_image[prop.image]
    res["std_intensity"] = np.std(intensities)

    percentiles = range(10, 100, 10)
    values = np.percentile(intensities, percentiles)
    for perc, val in zip(percentiles, values):
        res["perc_" + str(perc) + "_intensity"] = val

    if include(features_type, 'default') or include(features_type, 'perc_per_10') or include(features_type, 'mini'):
        res["median_intensity"] = np.median(intensities)
        res["perc_25_intensity"] = np.percentile(intensities, 25)
        res["perc_75_intensity"] = np.percentile(intensities, 75)
        pass

    if include(features_type, "complex"):
        res["mad_intensity"] = np.median(np.abs(intensities - res['perc_50_intensity']))

    return res


def intensity_complex_features_from_props(prop):
    """
    Calculate complex intensity features for a single object.

    These may be scale variant so the input should be rescaled to normalized size.
    Args:
        prop: property object of the cell

    Returns:
        dictionary of features with values for this cell
    """
    res = OrderedDict()

    centroid = np.array(prop.centroid)
    weighted_centroid = np.array(prop.weighted_centroid) if prop.max_intensity != 0 else centroid
    displacement = weighted_centroid - centroid
    displacement_diameters = np.linalg.norm(displacement) / prop.equivalent_diameter

    major_axis_length = prop.major_axis_length if prop.major_axis_length != 0 else 1
    displacement_majors = np.linalg.norm(displacement) / major_axis_length
    res['mass_displace_in_diameters'] = displacement_diameters
    res['mass_displace_in_majors'] = displacement_majors

    for z in range(0, 4):
        for y in range(0, 4):
            for x in range(0, 4):
                moment_val = prop.weighted_moments_normalized[z][y][x]
                if z + y + x >= 2:
                    if np.isnan(moment_val):
                        if prop.max_intensity != 0:
                            print("Nan weighted normalized moment: ", z, y, x)
                        moment_val = 0
                    res["moment_normalized_{}_{}_{}".format(z, y, x)] = moment_val

    # Moments Hu are not implemented yet for 3d (however it should be possible).
    # Let's do it as a last resort.

    return res


def texture_complex_features_from_props(label, prop):
    """
    Calculate complex texture features for a single object.

    These may be scale variant so the input should be rescaled to normalized size.
    Args:
        label: original label of the object
        prop: property object of the cell

    Returns:
        dictionary of features with values for this cell
    """
    res = OrderedDict()

    # Haralick texture features:
    cell_mask = prop.image
    intensity_uint8 = img_as_ubyte(prop.intensity_image, force_copy=True)
    if intensity_uint8.shape[0] == 1:  # mahotas.haralick do not handle single slice 3d data
        intensity_uint8 = np.stack([intensity_uint8[0], intensity_uint8[0]])
        cell_mask = np.stack([cell_mask[0], cell_mask[0]])

    try:
        # Make sure that object interior is not ignored as other zeros.
        intensity_uint8[cell_mask] = np.maximum(intensity_uint8[cell_mask], 1)
        haralick_array = mahotas.features.haralick(intensity_uint8, distance=1, ignore_zeros=True, return_mean_ptp=True)
    except ValueError:  # it may also fail in some corner case shapes
        print("Haralick fail to compute for id:", label)
        haralick_array = np.zeros((26,), dtype=np.double)

    res.update(zip(HARALICK_AGGR_FEATURES, haralick_array))

    return res


def extract_cells_features(images_volume, labels_volume, features_type, voxel_size=DESIRED_VOXEL_SIZE):
    """
    Extract cell morphological and channel related features.
    Args:
        images_volume: S x Y x X volume for which we want to calculate features
            assumes that it is already scaled to 0-1 range
        labels_volume: S x Y x X
            cell level segmentation as cell labels
        features_type: name of the features to calculate
        voxel_size: tuple with the real world size of the voxel (z,y,x), especially important to normalize Z axis which usually
            has much lower resolution (voxel is large in Z axis)
    Returns:
        dictionary of cells to cell features
    """
    res = {}

    props = measure.regionprops(label_image=labels_volume, intensity_image=images_volume)
    for prop in props:
        i = prop.label

        my_features = intensity_features_from_props(prop, features_type)
        if include(features_type, "complex") or include(features_type, "texture"):
            # rescale cell to standard size
            resampled_cell_image, actual_pixel_size = resample(prop.intensity_image, voxel_size, DESIRED_VOXEL_SIZE)
            resampled_cell_label, _ = resample(prop.image.astype(np.uint8), voxel_size, DESIRED_VOXEL_SIZE)
            if actual_pixel_size is not None:
                resample_props = measure.regionprops(label_image=resampled_cell_label,
                                                     intensity_image=resampled_cell_image)
                resample_prop = resample_props[0]
                assert len(resample_props) == 1, "More than one object in single object view."
            else:
                resample_prop = prop

            if include(features_type, "complex"):
                complex_features = intensity_complex_features_from_props(resample_prop)
                my_features.update(complex_features)

            if include(features_type, "texture"):
                texture_features = texture_complex_features_from_props(prop.label, resample_prop)
                my_features.update(texture_features)

        my_features['id'] = i
        res[i] = my_features
        prop._cache.clear()

    return res


def calc_safe_solidity(cell_prop, cell_size_z):
    """
    Calculate solidity of the given cell.

    It wraps standard calculation so that it is able to handle corner cases.
    Args:
        cell_prop: properties object of the 3D cell
        cell_size_z: size of the cell in 3D

    Returns:
        solidity: ratio of the cell area to area of convex of the cell
    """
    try:
        if cell_size_z == 1:
            # same way as in _regionprops.py
            from skimage.morphology.convex_hull import convex_hull_image
            return cell_prop.area / np.sum(convex_hull_image(cell_prop.image[0]))
        else:
            return cell_prop.solidity
    except QhullError:
        # did not manage to calculate, probably it is solid
        return 1.0


def extract_shape_features(labels_volume, features_type, voxel_size=DESIRED_VOXEL_SIZE):
    """
    Extract shape related features.
    Args:
        labels_volume: S x Y x X
            cell level segmentation as cell labels
        features_type: name of the features to calculate
        voxel_size: tuple with the real world size of the voxel (z,y,x), especially important to normalize Z axis which usually
            has much lower resolution (voxel is large in Z axis)

    Returns:
        dictionary of cells to cell features
    """
    res = {}

    props = measure.regionprops(label_image=labels_volume, intensity_image=labels_volume > 0)
    for prop in props:
        # If there is only one page z is not calculated.
        obj_image, actual_pixel_size = resample(prop.image.astype(np.uint8), voxel_size, DESIRED_VOXEL_SIZE)
        if actual_pixel_size is not None:
            resample_props = measure.regionprops(label_image=obj_image, intensity_image=obj_image > 0)
            resample_prop = resample_props[0]
            assert len(resample_props) == 1, "More than one object in single object view."
        else:
            resample_prop = prop

        try:
            z, y, x = prop.centroid
            sz, sy, sx, ez, ey, ex = prop.bbox
            size_z = ez - sz
        except ValueError:
            z = 0
            y, x = prop.centroid
            size_z = 1
        area = prop.area

        assert all(np.array(DESIRED_VOXEL_SIZE) == DESIRED_VOXEL_SIZE[0]), "Voxel size is not isotropic."
        desired_voxel_length = DESIRED_VOXEL_SIZE[0]

        my_features = {'id': prop.label,
                       'pos_z': z, 'pos_y': y, 'pos_x': x,
                       'area': area, 'size_z': size_z,
                       'volume_um': resample_prop.area * DESIRED_VOXEL_UM,
                       }

        if include(features_type, "complex"):
            solidity = calc_safe_solidity(resample_prop, size_z)

            eigen_values = np.array(resample_prop.inertia_tensor_eigvals)
            machine_errors = np.bitwise_and(eigen_values < 0, eigen_values > -1e-9)
            if np.sum(machine_errors) > 0:
                print('Encountered %d eigenvalues < 0 and > -1e-9, rounding to 0',
                      np.sum(machine_errors))
                eigen_values[machine_errors] = 0

            l1, l2, l3 = eigen_values
            first_axis_ratio = 0
            second_axis_ratio = 0
            if l2 != 0:
                first_axis_ratio = math.sqrt(1 - l3 / l2)
            if l1 != 0:
                second_axis_ratio = math.sqrt(1 - l2 / l1)

            if l3 < 0:
                raise Exception("Inertia eigenvalue for", prop.label, "lower than 0: ", l3)

            complex_shape = {
                'solidity': solidity,
                'first_major_diff': first_axis_ratio,
                'second_major_diff': second_axis_ratio,
                'inertia_length_0': 4 * math.sqrt(max(0, l1)) * desired_voxel_length,
                'inertia_length_1': 4 * math.sqrt(max(0, l2)) * desired_voxel_length,
                'inertia_length_2': 4 * math.sqrt(max(0, l3)) * desired_voxel_length
            }
            my_features.update(complex_shape)

        res[prop.label] = my_features
        prop._cache.clear()

    return res


def extract_all_features(images_volume, labels_volume, channels_with_preprocessing_list=None, features_type='default',
                         only_for_labels=None, voxel_size=DESIRED_VOXEL_SIZE):
    """
    Extract cell morphological and channel related features.
    Args:
        images_volume: S x Y x X x C or S x Y x X
            with channel for which we want to calculate features
        labels_volume: S x Y x X
            cell level segmentation as cell labels
        channels_with_preprocessing_list: list of channels with optional preprocessings to use e.g. ["1","2"] or ["1-equal","1"]
        features_type: name of the features to calculate
        only_for_labels: list of label value for which to calculate features
            if None then all are calculated
        voxel_size: tuple with the real world size of the voxel (z,y,x), especially important to normalize Z axis which usually
            has much lower resolution (voxel is large in Z axis)
            if None then no voxel resizing should be done

    Returns:
        dictionary of cells with complete dictionary of cell features
    """
    voxel_size = voxel_size or DESIRED_VOXEL_SIZE

    channels_with_preprocessing_list = [
        "0"] if channels_with_preprocessing_list is None else channels_with_preprocessing_list
    assert isinstance(channels_with_preprocessing_list, list) or isinstance(channels_with_preprocessing_list,
                                                                            np.ndarray)
    assert channels_with_preprocessing_list == ["0"] or images_volume.ndim > 3

    if only_for_labels is not None:
        labels_volume = replace_values(labels_volume, {i: i for i in only_for_labels}, zero_unmapped=True)

    # Calculate shape related features.
    res = {}
    shape_features = extract_shape_features(labels_volume, features_type, voxel_size)
    add_data_with_prefix(res, shape_features, '')

    # Rescale and pick only important channels.
    for channel_with_preprocess in tqdm(channels_with_preprocessing_list):
        processed_channel_volume, processed_labels_volume = preprocess_input_labels(images_volume, labels_volume,
                                                                                    channel_with_preprocess,
                                                                                    voxel_size)

        channel_features = extract_cells_features(processed_channel_volume, processed_labels_volume, features_type,
                                                  voxel_size)
        add_data_with_prefix(res, channel_features, str(channel_with_preprocess))
    return res
