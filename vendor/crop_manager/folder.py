import logging
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Set
from image import Image

import daiquiri
from PySide2 import QtWidgets


daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


class Folder:
    """
    category: `input` or `contour`
    class_: `negative` or `positive` or `uncertain`
    """

    mapping = {"negative": "0_negative", "positive": "1_positive", "uncertain": "2_uncertain"}

    def __init__(self, root: Path):
        logger.info(f"Root folder opened: {root}")
        self.root = root

        self.validate_presence()

    @classmethod
    def from_dialog(cls):
        path = Path(QtWidgets.QFileDialog.getExistingDirectory())
        return cls(path)

    def validate_presence(self):
        for class_folder in self.mapping.values():
            folder = self.root / class_folder
            if not folder.exists():
                raise ValueError(f"Folder not found: {folder}")

    def get_path_to_class_folder(self, class_: str) -> Path:
        return self.root / self.mapping[class_]

    def get_images(self):
        images = OrderedDict()
        for class_ in self.mapping.keys():
            image_pairs = self._get_image_pairs(class_)
            for image_root_name, categories in image_pairs.items():
                try:
                    images[image_root_name] = Image(
                        root_folder=self.root,
                        class_folder=self.mapping[class_],
                        name=image_root_name,
                        categories=categories,
                    )
                except AssertionError as e:
                    logger.warn(f"Missing contour or input file for: {image_root_name}")
        logger.info(f"Found {len(images)} image paris.")
        return images

    @classmethod
    def get_class_from_subfolder(cls, subfolder: str):
        """Return `negative` etc """
        for class_ in cls.mapping:
            if cls.mapping[class_] == subfolder:
                return class_
        raise KeyError(f"Given subfolder not listed in mapping.")

    @classmethod
    def get_subfolder_from_class(cls, class_: str):
        """Return `0_negative` etc """
        return cls.mapping[class_]

    def _get_image_pairs(self, class_: str, extension: str = "tif") -> Dict[str, Set[str]]:
        """Fetch `contour` and `input` images.
        Args:
            class_ (str): name of class e.g: `negative`

        Return:
            Image pairs(dict): key - root image name(trimmed '.contour.tif / .input.tif'), value - set of categories
        """
        image_pair = defaultdict(set)
        for file_ in self.get_path_to_class_folder(class_).glob(f"*.{extension}"):
            image_pair[self._get_name(file_)].add(self._get_category(file_))
        return image_pair

    def _get_name(self, file_path: Path) -> str:
        splitted = file_path.name.split(".")
        if len(splitted) < 2:
            raise ValueError(f"Invalid file name: {file_path}")
        return ".".join(splitted[:-2])

    def _get_category(self, file_path: Path):
        splitted = file_path.name.split(".")
        if splitted[-2] == "input":
            return "input"
        elif splitted[-2] == "contour":
            return "contour"
        else:
            raise ValueError(f"Invalid file name: {file_path}")
