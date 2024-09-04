import numpy as np

from clb.classify.instance_matching import cell_level_from_contours


def unique_cells(labels):
    return [x for x in np.unique(labels) if x > 0]


def cell_stats(segmentation, classes=None):
    """
    Calculate statistics from segmentation and optionaly predicted classes.
    Args:
        segmentation: S x Y x X volume of cell labels
        classes: S x Y x X volume of probabilities of cell belonging to the class

    Returns:
        dictionary with overall and per slice statistics
    """
    cell_labels = unique_cells(segmentation)
    chosen_cells = np.array(cell_labels)

    if classes is not None:  # get labels of class
        cell_classes = cell_level_from_contours(segmentation, classes)
        chosen_cells = [i for i in chosen_cells if cell_classes[i]['class'] > 0.5]

    stats_per_slice = []
    for slice_num in range(len(segmentation)):
        slice_cells = unique_cells(segmentation[slice_num])
        stats_of_slice = {"cell_number": len(slice_cells)}

        if classes is not None:
            chosen_slice_cells = list(set(slice_cells) & set(chosen_cells))
            stats_of_slice["class_cell_number"] = len(chosen_slice_cells)

        stats_per_slice.append(stats_of_slice)

    res = {"cell_number": len(cell_labels), "slices": stats_per_slice}

    if classes is not None:
        res["class_cell_number"] = len(chosen_cells)

    return res
