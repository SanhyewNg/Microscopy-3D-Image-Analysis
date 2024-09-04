import argparse

import numpy as np

import clb.dataprep.imaris.ims_file as ims_file
import clb.dataprep.readers as readers


class ChannelParser(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        channel_infos = [parse_channel(channel) for channel in values]
        setattr(namespace, self.dest, channel_infos)


def parse_channel(channel):
    """Parse channel description into dictionary.

    Args:
        channel (str): Channel description in format <path>,<name>,[channel],[color].

    Returns:
        dict: Channel information, keys: path, name, channel, color.
    """
    info_pieces = channel.split(',')
    if len(info_pieces) < 2 or len(info_pieces) > 4:
        raise ValueError('Wrong format of channel description: {}. '
                         'Should be <path>,<name>,[channel],[color]'.format(channel))

    channel_info = {
        'path': info_pieces[0],
        'name': info_pieces[1],
        'channel': 0 if len(info_pieces) < 3 else int(info_pieces[2]),
        'color': None if len(info_pieces) < 4 else info_pieces[3],
    }

    return channel_info


def add_channel(fileobj, channel_info):
    """Add channel described by `channel_info` to `fileobj`.

    Args:
        fileobj (ImsFile): File to add channel to.
        channel_info (dict): Channel information same form as `parse_channel` return
                             value.
    """
    path = channel_info['path']
    channel = channel_info['channel']
    data = np.squeeze(readers.get_volume_reader(path)[:, channel])
    color_mode = 'TableColor' if channel_info['color'] is None else 'BaseColor'
    color_value = None if color_mode == 'TableColor' else channel_info['color']
    fileobj.add_channel(data=data, color_mode=color_mode, color_value=color_value)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path')
    parser.add_argument('--inputs', nargs='+', action=ChannelParser)
    return parser


def main(output_path, inputs):
    with ims_file.ImsFile(output_path, mode='x') as f:
        for channel_info in inputs:
            add_channel(f, channel_info)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
