import fire

from clb.dataprep.imaris.ims_file import (add_channel_from_tif,
                                          extract_channels_to_tif)


class ImsEditor:
    @staticmethod
    def extract(ims_path, tif_path, channels=-1):
        """Extract channels from ims file to tif file.

        Args:
            ims_path (str): Path to ims file to extract channels from.
            channels (list): Channels to extract.
            tif_path (str): Path to tif file to save channels to.
        """
        extract_channels_to_tif(ims_path, channels, tif_path)

    @staticmethod
    def add(ims_path, tif_path, channel, color, name):
        """Add channels to ims file.

        Args:
            ims_path (str): Path to ims file.
            tif_path (str): Path to tif file.
            channel (int): Index of channel in tif file.
            color (str): Channel color.
            name (str): Channel name.
        """
        add_channel_from_tif(ims_path, tif_path, color, name, channel)

    @staticmethod
    def add_labels(ims_path, tif_path, name):
        """Add labels as multicolor channel to ims file.

        Args:
            ims_path (str): Path to ims file.
            tif_path (str): Path to tif file.
            name (str): Channel name.
        """
        add_channel_from_tif(ims_path, tif_path, "Segmentation", name, 0)


if __name__ == '__main__':
    fire.Fire(ImsEditor)
