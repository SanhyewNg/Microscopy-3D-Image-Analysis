import logging
import os
import random
import sys
from collections import OrderedDict, defaultdict
from functools import partial
from pathlib import Path
from random import randint
from shutil import move
from typing import Any, Callable, Dict, List, Tuple, Union

import daiquiri
import imageio
import matplotlib
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import arange, pi, sin
from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt

from filters import FILTER_MAPPING, FilterDialog
from folder import Folder
from image import Image
from tsv_utils import CellData, ProcessData

matplotlib.use("Qt5Agg")


daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

progname = os.path.basename(sys.argv[0])
progversion = "0.7"


def show_message(message, parent=None):
    dial = QtWidgets.QMessageBox(parent)
    dial.setText(message)
    dial.exec_()

class Canvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)
        self.image = None
        self.title = ""

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, image, z=0, title=""):
        self.title = title
        self.axes.cla()
        self.image = image
        self.axes.set_title(self.title)
        if self.image.ndim == 4:
            self.axes.imshow(self.image[z, :, :, :])
        else:
            self.axes.imshow(self.image[z, ...])
        self.draw()

    def update_z(self, z=0):
        self.axes.cla()
        self.axes.set_title(self.title)
        if self.image.ndim == 4:
            self.axes.imshow(self.image[z, :, :, :])
        else:
            self.axes.imshow(self.image[z, ...])
        self.draw()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Crop Manager")

        self.image = None  # current image
        self.images = None  # all images
        self.filtered = None  # image keys that should be shown
        self.processed = None  # info on which images are already done
        self.cell_data = None  # TSV information
        self.enabled_filters = ["seen"]

        self.create_menu()
        self.create_slider()

        self.main_widget = QtWidgets.QWidget(self)
        l = QtWidgets.QVBoxLayout(self.main_widget)
        self.cc = Canvas(self.main_widget, width=5, height=4, dpi=100)
        self.ic = Canvas(self.main_widget, width=5, height=4, dpi=100)

        buttons = QtWidgets.QHBoxLayout()
        self.positive_btn = QtWidgets.QPushButton("Positive")
        self.negative_btn = QtWidgets.QPushButton("Negative")
        self.uncertain_btn = QtWidgets.QPushButton("Uncertain")
        buttons.addWidget(self.positive_btn)
        buttons.addWidget(self.negative_btn)
        buttons.addWidget(self.uncertain_btn)
        self.positive_btn.clicked.connect(partial(self.move_image_to, "positive"))
        self.negative_btn.clicked.connect(partial(self.move_image_to, "negative"))
        self.uncertain_btn.clicked.connect(partial(self.move_image_to, "uncertain"))

        images_layout = QtWidgets.QHBoxLayout()
        images_layout.addWidget(self.ic)
        images_layout.addWidget(self.cc)

        l.addLayout(images_layout)
        l.addLayout(buttons)
        l.addWidget(self.slider)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def move_image_to(self, dest_class):
        dest = self.root_folder.get_path_to_class_folder(dest_class)
        self.image.move_to(dest)
        self.image.class_folder = Folder.get_subfolder_from_class(dest_class)
        self.processed.update(self.image.name, dest_class)
        self.next_image()

    def open_filter_dialog(self):
        self.filter_dialog = FilterDialog(self, self.enabled_filters)
        self.filter_dialog.exec_()
        self.enabled_filters = self.filter_dialog.get_enabled_filters()
        self.apply_filters()

    def apply_filters(self):
        preserved = set(self.images.keys())

        for filter_ in self.enabled_filters:
            preserved = preserved & FILTER_MAPPING[filter_](self.images, self.processed, self.cell_data)
        logger.info(f"Filtered out to: {len(preserved)} images")
        self.filtered = list(preserved)

    def next_image(self):
        try:
            index = randint(0, len(self.filtered))
            image_root_name = self.filtered.pop(index)  # list of image_root_name that could be shown
        except IndexError:
            logger.info("No more images to process")
            show_message("No more images to process. Closing...", self)
            self.fileQuit()
        self.image = self.images[image_root_name]
        images = self.image.read()
        names = self.image.full_name
        self.ic.plot(images["input"], self.slider.value(), names["input"])
        self.cc.plot(images["contour"], self.slider.value(), names["contour"])

    def slider_change(self):
        slider_value = self.slider.value()
        self.ic.update_z(slider_value)
        self.cc.update_z(slider_value)

    def fileQuit(self):
        self.processed.__exit__()
        self.close()
        sys.exit(0)

    def file_open_and_load_data(self):
        self.root_folder = Folder.from_dialog()
        self.images = self.root_folder.get_images()
        self.processed = ProcessData(self.root_folder.root / "process_info.tsv")
        self.processed.__enter__()
        self.apply_filters()
        self.next_image()

    def create_class_tsvs(self):
        subfolders = {"0_negative": [], "1_positive": [], "2_uncertain": []}
        for image in self.images.values():
            subfolders[image.class_folder].append(image)
        for subfolder in subfolders:
            logger.info(f"Generating TSV for {subfolder}. It may take a while ...")
            self.cell_data.generate_tsv(subfolders[subfolder], self.root_folder.root / subfolder / "features.tsv")

    def file_open_tsv_and_load_data(self):
        self.cell_data = CellData.from_dialog()
        self.cell_data.validate_sync(
            {name: Folder.get_class_from_subfolder(img.class_folder) for (name, img) in self.images.items()}
        )

    def update_tsv(self):
        self.cell_data.update_class(
            {name: Folder.get_class_from_subfolder(img.class_folder) for (name, img) in self.images.items()}
        )

    def closeEvent(self, ce):
        self.fileQuit()

    def create_menu(self):
        self.file_menu = QtWidgets.QMenu("&File", self)
        self.file_menu.addAction("&Quit", self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.file_menu.addAction("&Open Folder", self.file_open_and_load_data, QtCore.Qt.CTRL + QtCore.Qt.Key_O)

        self.filter_menu = QtWidgets.QMenu("&Filter", self)
        self.filter_menu.addAction("Open filters", self.open_filter_dialog)

        self.tsv_menu = QtWidgets.QMenu("&TSV", self)
        self.tsv_menu.addAction("&Load Cell TSV", self.file_open_tsv_and_load_data, QtCore.Qt.CTRL + QtCore.Qt.Key_T)
        self.tsv_menu.addAction("Update Cell TSV", self.update_tsv)
        self.tsv_menu.addAction("Export class TSV-s", self.create_class_tsvs)

        self.menuBar().addMenu(self.file_menu)
        self.menuBar().addMenu(self.filter_menu)
        self.menuBar().addMenu(self.tsv_menu)

    def create_slider(self):
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setMaximum(31)
        self.slider.valueChanged.connect(self.slider_change)


if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle(f"{progname}")
    aw.show()
    sys.exit(qApp.exec_())
