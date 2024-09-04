"""Module defines general tools that are used in statistics calculation."""
import functools as ft
import itertools as it
import operator as op

import numpy as np
import pandas as pd
import skimage
from typing import Set

import clb.classify.instance_matching as im
from clb.classify.visualisation import ClassificationVolume


def insert_classes(labels_volume, stats_df, **classes_to_volumes):
    """Insert column with classes groups at the beginning of dataframe.

    For cells with no class 'no_class' string is inserted.

    Args:
        labels_volume (np.ndarray): Volume with cell labels.
        stats_df (pd.DataFrame): DataFrame with statistics.
        classes_to_volumes: Mapping class names to VolumeIters.
    """
    all_ids = get_all_ids(labels_volume)
    classes_to_ids = map_classes_to_ids(labels_volume,
                                        **classes_to_volumes)
    if '' in classes_to_ids:
        classes_to_ids['no_class'] = classes_to_ids.pop('')
    ids_to_class_groups = map_ids_to_class_groups(all_ids,
                                                  classes_to_ids)
    class_column = pd.Series(ids_to_class_groups).replace(to_replace='', value='no_class')
    stats_df.insert(loc=0, column='classes', value=class_column)


def get_class_ids(labels: np.ndarray, classes: np.ndarray) -> Set[int]:
    """Get ids of cells that are classified positive.

    Args:
        labels (np.ndarray): Volume with cells.
        classes (np.ndarray): ClassificationVolume (binary) .

    Returns:
        set: Ids of cells that are classified positive.
    """
    id_to_classes = im.cell_level_from_contours(labels, classes)
    ids = {cell_id for cell_id, class_info in id_to_classes.items() if class_info['class'] > 0}
    return ids

def get_all_ids(labels):
    """Return all different cell ids in `labels`.

    Args:
        labels (np.ndarray): Volume with cell labels.

    Returns:
        set: Different ids.
    """
    all_ids = set(np.unique(labels)) - {0}

    return all_ids


def map_classes_to_ids(labels, **classes):
    """Return mapping from class name to ids of cells in it.

    Classes names are used as dictionary keys.

    Args:
        labels (np.ndarray): Volume with labels.
        classes: VolumeIters with classification results.

    Returns:
        dict: Mapping classes to ids.
    """
    classes_to_ids = {
        name: get_class_ids(labels, ClassificationVolume.from_array(np.squeeze(class_volume), binary=True))
        for name, class_volume in classes.items()
    }

    return classes_to_ids


def map_ids_to_class_groups(all_ids, classes_to_ids):
    """Map cell ids to class groups.

    Args:
        all_ids (set): Ids of all cells.
        classes_to_ids (dict): Class names to cell ids.

    Returns:
        dict: Cell ids to class groups.
    """
    ids_to_groups = {cell_id: [] for cell_id in all_ids}

    for name, ids in classes_to_ids.items():
        for cell_id in ids:
            ids_to_groups[cell_id].append(name)

    for cell_id, names_list in ids_to_groups.items():
        ids_to_groups[cell_id] = ', '.join(sorted(names_list))

    return ids_to_groups


def find_ids_of_classes_cells(names_to_ids, names):
    """Find ids of cells that belong to all classes with given `names`.

    Args:
        names_to_ids (dict): Mapping class name to sets of ids of cells.
        names (Iterable): Names of classes.

    Returns:
        set: Ids of cells that belong to all classes with `names`.
    """
    id_sets = (names_to_ids[name] for name in names)
    all_class_ids = ft.reduce(op.and_, id_sets)

    return all_class_ids


def map_class_groups_to_ids(all_ids, classes_to_ids):
    """Return mapping combinations of classes names to ids of cells.

    Keys are created from names of classes separated by commas. There are two
    additional keys: 'all_cells' and 'no_class'. So if there are two classes
    'a' and 'b' there will be five keys:
    - 'all_cells'
    - 'no_class'
    - 'a'
    - 'b'
    - 'a, b'.

    Args:
        all_ids (set): All cell ids.
        classes_to_ids (dict): Mapping name of class to ids of cells.

    Returns:
        dict: Mapping combinations of classes to ids of cells.
    """
    names_groups = (it.combinations(sorted(classes_to_ids.keys()), r)
                    for r in range(1, len(classes_to_ids) + 1))
    names_groups = it.chain.from_iterable(names_groups)
    class_groups_to_ids = {
        ', '.join(names): find_ids_of_classes_cells(classes_to_ids, names)
        for names in names_groups
    }
    any_class_ids = ft.reduce(op.or_, classes_to_ids.values())
    no_class_ids = all_ids - any_class_ids
    class_groups_to_ids['all_cells'] = all_ids
    class_groups_to_ids['no_class'] = no_class_ids

    return class_groups_to_ids
