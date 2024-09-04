import os
from datetime import datetime

import fire
import imageio
import numpy as np
import pandas
from tqdm import tqdm

from skimage import img_as_ubyte

# clb import required to subscribe LIF reader
# noinspection PyUnresolvedReferences
import clb


def summary(input_path, output):
    """Extract summary from all series of LIF file.
    Get example slices and metadata formatted for easy copy do Databases spreadsheet.

    Usage:
        lif_summary.py "...\Public\new_data.lif" "...\Public\Datasets summary\181023 new_data"
        It would creates:
        - "181023 new_data.tif" file with sample slice from each series
        - "181023 new_data.tsv" file with dimensions and voxel size data which can then be copied

    Args:
        input_path (str): Path to input LIF file.
        output (str): Template path for saving output summary. TSV and multi-page TIF file will be created.
    """
    series_example_slices = []
    series_example_metadata = []
    lif_name = os.path.splitext(os.path.basename(input_path))[0]
    tif_output_path = output + ".tif"
    tsv_output_path = output + ".tsv"

    with imageio.get_reader(input_path, channels=0) as reader:
        for i in tqdm(range(0, reader.get_series_num())):
            with imageio.get_reader(input_path, series=i) as series_reader:
                number_of_slices = series_reader.get_length()
                input_arr = series_reader.get_data(number_of_slices // 3, channels=None)
                for c in range(series_reader.size_c()):
                    input_arr_channel = img_as_ubyte(input_arr[..., c])
                    image_metadata_channel = series_reader.get_meta_data(0)['Channels'][c]

                    color = clb.dataprep.lif.utils.decode_color(int(image_metadata_channel['Color']))
                    rgb = []
                    for k in range(3):
                        rgb.append(color[k] / 255.0 * input_arr_channel)
                    rgb_array = np.stack(rgb, axis=-1)
                    series_example_slices.append(rgb_array.astype(np.uint8))

                image_metadata = series_reader.get_meta_data(0)
                image_metadata["filename"] = lif_name
                image_metadata["series_name"] = series_reader.get_name() + ": S" + str(i)
                print("Processing", "'" + image_metadata["series_name"] + "'", "...")

                image_metadata["resolution_x"] = input_arr.shape[1]
                image_metadata["resolution_y"] = input_arr.shape[0]
                image_metadata["resolution_z"] = number_of_slices
                image_metadata["sample_size_x"] = image_metadata["resolution_x"] * float(image_metadata['PhysicalSizeX'])
                image_metadata["sample_size_y"] = image_metadata["resolution_y"] * float(image_metadata['PhysicalSizeY'])
                image_metadata["sample_size_z"] = image_metadata["resolution_z"] * float(image_metadata['PhysicalSizeZ'])

                image_metadata["imaging parameters"] = "{0}x{1}x{2}".format(image_metadata["resolution_x"],
                                                                            image_metadata["resolution_y"],
                                                                            image_metadata["resolution_z"])
                image_metadata["voxel size"] = ("{0:.2f}x\n" +
                                                "{1:.2f}x\n" +
                                                "{2:.2f}um\n" +
                                                "\n" +
                                                "{3:.0f}x\n" +
                                                "{4:.0f}x\n" +
                                                "{5:.0f}um").format(float(image_metadata["PhysicalSizeX"]),
                                                                    float(image_metadata["PhysicalSizeY"]),
                                                                    float(image_metadata["PhysicalSizeZ"]),
                                                                    image_metadata["sample_size_x"],
                                                                    image_metadata["sample_size_y"],
                                                                    image_metadata["sample_size_z"]
                                                                    )

                modification_time = os.path.getmtime(input_path)
                image_metadata["date modification"] = datetime.fromtimestamp(modification_time).strftime("%m/%d/%Y")

                series_example_metadata.append(image_metadata)

    imageio.mimwrite(tif_output_path, series_example_slices)
    important_columns = ["filename", "series_name", "imaging parameters", "voxel size", "date modification"]

    df = pandas.DataFrame(series_example_metadata)
    df.to_csv(tsv_output_path, columns=important_columns, sep='\t')


if __name__ == '__main__':
    fire.Fire(summary)
