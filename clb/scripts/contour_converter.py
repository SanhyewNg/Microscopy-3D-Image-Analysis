import logging
from pathlib import Path

import daiquiri
import imageio
from fire import Fire
from tqdm import tqdm

from clb.image_processing import enhance_contour_visibility

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


def improve_contours_in_folders(folder_path):
    files = Path(folder_path).glob("**/*.contour.tif")
    for contour_file in tqdm(files):
        contour_file = str(contour_file)
        target_cell_value = contour_file.split(".")[-3]
        image = imageio.volread(contour_file)
        try:
            enhanced = enhance_contour_visibility(image, int(target_cell_value))
            imageio.mimwrite(contour_file, list(enhanced))
        except ValueError:
            logger.warn("Exception during improvement of {}".format(contour_file))


if __name__ == "__main__":
    Fire(improve_contours_in_folders)
