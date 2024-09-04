import logging
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Set, Tuple, Union
import tqdm

import daiquiri
import pandas as pd
from PySide2 import QtWidgets

from image import Image

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


class CellData:
    mapping = {"negative": 0, "positive": 1, "uncertain": 2}

    def __init__(self, path: Path):
        self.path = path
        self.data = pd.read_csv(self.path, sep="\t", index_col=[0, 1])
        logger.info("Created cell data instance")

    @classmethod
    def from_dialog(cls):
        path = Path(QtWidgets.QFileDialog.getOpenFileName(filter="TSV Files (*.tsv)")[0])
        logger.info(f"Opening Cell Data file: {path}")
        return cls(path)

    def update_class(self, images):
        """Update class on of given images."""
        logger.info(f"Updating class values in {self.path}. It may take a while...")
        for image_root_name, class_no in tqdm.tqdm(images.items()):
            idx = image_root_name.rsplit(".", 1)
            idx[1] = int(idx[1])
            idx = tuple(idx)
            try:
                self.data.loc[idx, "class"] = class_no
            except KeyError:
                logger.warn(f"Image {image_root_name} not present in TSV, removing from tsv...")
                self.data.drop(index=idx, inplace=True)
        self.save()

    def generate_tsv(self, images: List[Image], path: Path):
        """Create `features.tsv` to certain path."""
        df = self._create_empty_dataframe()
        for image in tqdm.tqdm(images):
            try:
                row = self.data.loc[(image.src_name, image.cell_number)]
                df = df.append(row)
            except KeyError:
                logger.warn(f"Image {image.name} not present in TSV")
        self.save(df, path)

    def save(self, dataframe: pd.DataFrame = None, path: Path = None):
        path = path or self.path
        dataframe = dataframe if dataframe is not None else self.data
        dataframe.to_csv(path, sep="\t")
        logger.info(f"CellData saved to TSV: {path}")

    def _create_empty_dataframe(self):
        multi_index = pd.MultiIndex(levels=[[], []], codes=[[], []])
        return pd.DataFrame(index=multi_index, columns=self.data.columns.values)

    def validate_sync(self, images):
        """Check if all images are in correct subfolders.
        
        Args: 
            images (dict): {root_image_name (str): 0 (class - int)}
        """
        logger.info("Validating sync between TSV and folders...")
        incorrect_class_folder = []
        images_not_in_data = []
        indexes = tuple(self.data.index)
        for image_root_name, class_no in images.items():
            idx = tuple(image_root_name.rsplit(".", 1))
            if idx not in indexes:
                images_not_in_data.append(image_root_name)
            elif self.data.loc[idx, "class"] != class_no:
                incorrect_class_folder.append(image_root_name)
        if incorrect_class_folder:
            logger.warn(
                f"Detected images located in class folders incorrect to TSV:\n{pformat(incorrect_class_folder)}"
            )
        if images_not_in_data:
            logger.warn(f"Detected images that are not listed TSV:\n{pformat(images_not_in_data)}")
        # TODO: add missing data to images


class ProcessData:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        try:
            logger.info(f"Opening process data file: {self.path}")
            self.info = pd.read_csv(self.path, sep="\t", index_col=0)
        except FileNotFoundError:
            logger.info(f"Process data file not found: {self.path}, creating new one...")
            self.info = pd.DataFrame(columns=["class"])

    def __exit__(self):
        if self.info is not None:
            logger.info(f"Saving process data file to: {self.path}")
            self.info.to_csv(self.path, sep="\t")
        else:
            logger.info(f"No data to save in process data file: {self.path}")

    def is_done(self, image: Image) -> bool:
        return image.name in self.info.index

    def update(self, image_root_name, dest_class):
        self.info.loc[image_root_name, "class"] = dest_class
