import logging
from collections import defaultdict
from functools import partial
from pathlib import Path
from shutil import move
from typing import Any, Callable, Dict, List, Tuple, Union

import daiquiri
import imageio
import matplotlib
from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import QObject, Qt, Signal, Slot

from folder import Folder
from image import Image
from tsv_utils import CellData, ProcessData

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


def filter_seen(images, processed=None, cell_data=None):
    images_indexes = set(images.keys())
    processed_indexes = set(processed.info.index.values)
    difference = images_indexes.difference(processed_indexes)
    logger.info(f"{len(difference)}, {len(images_indexes)}, {len(processed_indexes)}")
    return difference


def filter_tsv(images, processed=None, cell_data=None):
    images_indexes = set(images.keys())
    try:
        cell_data_indexes = set([f"{n}.{i}" for (n, i) in cell_data.data.index.values])
    except AttributeError:
        logger.warn("Cannot filter by TSV. TSV is loaded ? ")
        return set()
    return images_indexes & cell_data_indexes


def filter_zero_class_fraction(images, processed=None, cell_data=None):
    logger.error("Not implemented")
    exit(123)


class FilterClass:
    def __init__(self, class_):
        self.class_ = class_

    def __call__(self, images, processed=None, cell_data=None):
        return set(
            [name for (name, img) in images.items() if Folder.get_class_from_subfolder(img.class_folder) == self.class_]
        )


class FilterDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, enabled_filters=()):
        super(FilterDialog, self).__init__(parent)
        self.enabled_filters = enabled_filters

        self.setWindowTitle("Filters")

        filter_seen = QtWidgets.QCheckBox("Filter seen")
        filter_seen.setChecked("seen" in self.enabled_filters)

        filter_negative = QtWidgets.QCheckBox("Filter to Negative")
        filter_negative.setChecked("negative" in self.enabled_filters)

        filter_positive = QtWidgets.QCheckBox("Filter to Positive")
        filter_positive.setChecked("positive" in self.enabled_filters)

        filter_uncertain = QtWidgets.QCheckBox("Filter to Uncertain")
        filter_uncertain.setChecked("uncertain" in self.enabled_filters)

        filter_zero_class_fraction = QtWidgets.QCheckBox("Filter zero Class fraction")
        filter_zero_class_fraction.setChecked("zero_class_fraction" in self.enabled_filters)

        filter_by_tsv = QtWidgets.QCheckBox("Filter by TSV")
        filter_by_tsv.setChecked("tsv" in self.enabled_filters)

        self.buttons = {
            "seen": filter_seen,
            "negative": filter_negative,
            "positive": filter_positive,
            "uncertain": filter_uncertain,
            "zero_class_fraction": filter_zero_class_fraction,
            "tsv": filter_by_tsv,
        }

        buttonBox = QtWidgets.QDialogButtonBox(QtCore.Qt.Vertical)
        buttonBox.addButton(filter_seen, QtWidgets.QDialogButtonBox.ActionRole)
        buttonBox.addButton(filter_negative, QtWidgets.QDialogButtonBox.ActionRole)
        buttonBox.addButton(filter_positive, QtWidgets.QDialogButtonBox.ActionRole)
        buttonBox.addButton(filter_uncertain, QtWidgets.QDialogButtonBox.ActionRole)
        # buttonBox.addButton(filter_zero_class_fraction, QtWidgets.QDialogButtonBox.ActionRole)
        buttonBox.addButton(filter_by_tsv, QtWidgets.QDialogButtonBox.ActionRole)

        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(buttonBox)

        self.setLayout(mainLayout)

    def get_enabled_filters(self):
        return [k for (k, btn) in self.buttons.items() if getattr(btn, "isChecked")()]


FILTER_MAPPING = {
    "seen": filter_seen,
    "negative": FilterClass("negative"),
    "positive": FilterClass("positive"),
    "uncertain": FilterClass("uncertain"),
    "zero_class_fraction": filter_zero_class_fraction,
    "tsv": filter_tsv,
}
