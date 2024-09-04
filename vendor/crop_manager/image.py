import logging
from functools import partial
from pathlib import Path
from shutil import move
from typing import Any, Dict, List, Set, Tuple, Union

import daiquiri
import imageio
import matplotlib
import numpy as np

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


class Image:
    required = ("input", "contour")  # required image categories

    def __init__(self, root_folder: Path, class_folder: str, name: str, categories: Set[str]):
        self.root_path = root_folder
        self.class_folder = class_folder
        self.name = name  # TestNewClasses S1 1024 crop_class_pdlcd_10.tif.123
        self.cell_number = int(name.split(".")[-1])  # 123
        self.src_name = ".".join(name.split(".")[:-1])  # TestNewClasses S1 1024 crop_class_pdlcd_10.tif
        self.categories = {c: f".{c}.tif" for c in sorted(categories)}
        assert set(self.categories) == set(self.required), f"Not all required images are provided for {name}"

    def move_to(self, dest: Path):
        if self.class_path != dest:
            src = [str(self._get_full_path(c)) for c in self.categories.keys()]
            move_image = partial(move, dst=dest)
            call_each(move_image, src)
            logger.info(f"Moved {self.name} to {dest}.")

    def read(self) -> Dict[str, np.ndarray]:
        return {c: imageio.volread(self._get_full_path(c)) for c in self.categories}

    def get_full_name(self, category: str) -> str:
        return f"{self.name}{self.categories[category]}"

    @property
    def full_name(self) -> str:
        return {c: self.get_full_name(c) for c in self.categories}

    @property
    def class_path(self) -> Path:
        return self.root_path / self.class_folder

    @property
    def full_path(self):
        return {c: self._get_full_path(c) for c in self.categories}

    def _get_full_path(self, category) -> Path:
        return self.root_path / self.class_folder / self.get_full_name(category)

def call_each(func, items):
    for item in items:
        func(item)