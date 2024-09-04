import argparse
import copy
import os
import warnings

import imageio
import numpy as np
import pandas
import webcolors as wbcols

import clb.classify.feature_extractor as fe
import clb.classify.extractors as ex
import clb.classify.visualisation as vis
import clb.classify.predictors as predictors
import clb.dataprep.imaris.ims_file as ims_file
import clb.dataprep.readers as readers
import clb.dataprep.uff.uff_writers as uwriters
import clb.dataprep.utils as utils
import clb.run as run
import clb.yaml_utils as yutils
from clb import yaml_utils
from clb.classify.cell_extractor import extract_all_cells_crops


def ensure_labels(input, output, model, channel, series, start, stop):
    """Ensure segmentation labels are at `output`.

    If there is no file there segmentation is called.

    Args:
        input (str): Path to input volume.
        output (str): Path to file where labels should be.
        model (str): Path to instance segmentation model.
        channel (int|None): Channel to use during segmentation.
        series (int): Index of series to segment.
        start (int): First slice.
        stop (int|None): Slice after the last to segment.
    """
    if not os.path.isfile(output):
        run.make_instance_segmentation(input=input, model=model, outputs=[output],
                                       start=start, stop=stop, series=series,
                                       use_channel=channel)


def get_cell_crops(input_volume, labels_volume, channels, use_cubes, voxel_size=None):
    """Extract cube crop around each of the cells.

    Args:
        input_volume (array_like): Input volume.
        labels_volume (array_like): Volume with segmentation labels.
        channels (str): Channels with optional preprocessings to use 'e.g.
                        "1,2" or "1-equal,1".
        use_cubes (float): side of the cube around each cell in um
        voxel_size (tuple): voxel size of the input data, if None then no voxel resizing should be done

    Returns:
        dictionary of cells to cell crops in form of numpy volumes created after resizing to DESIRED_VOXEL_SIZE
    """
    input_volume = np.squeeze(input_volume)
    labels_volume = np.squeeze(labels_volume)
    return extract_all_cells_crops(input_volume, labels_volume, channels, use_cubes, voxel_size=voxel_size)


def get_features(input_volume, labels_volume, channels, features_type, voxel_size=None):
    """Calculate features of cells.

    Args:
        input_volume (array_like): Input volume.
        labels_volume (array_like): Volume with segmentation labels.
        channels (str): Channels with optional preprocessings to use 'e.g.
                        "1,2" or "1-equal,1".
        features_type (str): Feature type identifier.
        voxel_size: voxel size of the input data, if None then no voxel resizing should be done

    Returns:
        pandas.DataFrame: Calculated features.
    """
    input_volume = np.squeeze(input_volume)
    labels_volume = np.squeeze(labels_volume)
    channels = ex.parse_channels_preprocessing(channels)

    features_dict = fe.extract_all_features(input_volume,
                                            labels_volume,
                                            channels,
                                            features_type,
                                            voxel_size=voxel_size)
    features = pandas.DataFrame(list(features_dict.values()))
    features = features.set_index('id')
    features = features[fe.get_feature_columns(features.columns,
                                               features_type)]

    return features


def overlay_predictions(model_path, labels, features, cropped_cubes, discrete):
    """Predict classes of cells from given `features`.

    Args:
        model_path (str): Path to .pkl or .h5 classification model.
        labels (np.ndarray): Volume with labels.
        features (pandas.DataFrame): Features of cells.
        cropped_cubes (dict): Dictionary with cube around each cell.
        discrete (str): What kind of discretization to use. If None raw
                        probabilities are returned.

    Returns:
        np.ndarray: Classes overlayed on cells.
    """
    # TODO ensure both features and cropped_cubes have the same order if in future we can use both
    assert (cropped_cubes is not None) or (features is not None), "Neither features or cubes provided."
    assert (cropped_cubes is not None) ^ (features is not None), "Combining features and cubes is not yet supported."

    # TODO we may need to work lazily with crops as them may be bigger than entire volume
    predictor = predictors.load_predictor(model_path)

    index_list = []
    raw_crops = []
    if cropped_cubes is not None:
        index_list, crops = zip(*cropped_cubes.items())
        raw_crops = [crop['raw_input'] for crop in crops]

    if features is not None:
        index_list = features.index

    prediction = predictor.predict_discrete(features, raw_crops, discrete)
    id_to_classes = dict(zip(index_list, prediction))
    overlayed = vis.ClassificationVolume.create(labels_volume=labels,
                                                cell_classes=id_to_classes,
                                                rescale=False)

    return overlayed


def write_imaris(path,
                 dapi_volume,
                 labels,
                 marker_volume,
                 channels,
                 classes,
                 channel_name,
                 channel_color,
                 metadata):
    """Save classification results to .ims file.

    Args:
        path (str): Path to file.
        dapi_volume (array_like): Volume with dapi channel.
        labels (array_like): Segmentation labels, should be only one
                             channel.
        marker_volume (array_like): Volume used during classification.
        channels (list): Channels used for classification.
        classes (array_like): Classes overlayed on cells.
        channel_name (str): Name of channel with classes in .ims file.
        channel_color (str): Color of channel with classes in .ims file.
        metadata (dict): Image metadata.
    """
    images = [dapi_volume, labels, classes]
    dapi_volume, labels, classes = (np.squeeze(array) for array in images)
    marker_volume = np.asarray(marker_volume)
    if marker_volume.ndim < 4:
        marker_volume = marker_volume[..., np.newaxis]

    file_exists = os.path.isfile(path)
    mode = 'r+' if file_exists else 'x'
    with ims_file.ImsFile(path, mode=mode, image_metadata=metadata) as writer:
        if not file_exists:
            writer.add_channel(data=dapi_volume,
                               color_mode='BaseColor',
                               color_value='Blue',
                               channel_name='DAPI')
            writer.add_channel(data=labels,
                               color_mode='TableColor',
                               color_value='Blue',
                               channel_name='DAPI_labels')

        for channel in channels:
            name = channel_name + '_input_{}'.format(channel)
            writer.add_channel(data=marker_volume[..., channel],
                               color_mode='BaseColor',
                               color_value=channel_color,
                               channel_name=name)

        writer.add_channel(data=classes,
                           color_mode='BaseColor',
                           color_value=channel_color,
                           channel_name=channel_name)


def write_uff(path, metadata, classes, classif_name, colors):
    """Save classification results to .uff file.

    Args:
        path (str): Path to output file.
        metadata (dict): Metadata of the input file, described in uff.utils.
        classes (array_like): Classes overlayed on cells, should be only one
                              channel.
        classif_name (str): Name of classification.
        colors (tuple): Colors for negative and positive classes.
    """
    classes = utils.ensure_4d(classes)

    # Copying metadata, because we are going to change it later.
    metadata = copy.deepcopy(metadata)

    (metadata['SizeZ'], metadata['SizeY'], metadata['SizeX'],
     metadata['SizeC']) = map(str, utils.ensure_shape_4d(classes.shape))
    names_to_colors = {'negative': wbcols.name_to_hex(colors[0]),
                       'positive': wbcols.name_to_hex(colors[1])}
    metadata['Channels'] = [{'Name': 'classification {} {}'.
        format(classif_name, names_to_colors)}]

    if metadata.get("Name") is None:
        metadata['Name'] = os.path.basename(path)

    colors_palette = np.zeros((256, 4))
    # Negatives are marked with value 2, positives with 255.
    colors_palette[[2, 255]] = [(*wbcols.name_to_rgb(colors[0]), 255),
                                (*wbcols.name_to_rgb(colors[1]), 255)]

    writer = uwriters.UFFWriter(path, data=classes, metadata=metadata)
    writer.write(colors_palettes=[colors_palette])


def get_parser():
    parser = argparse.ArgumentParser(description='CLB classification.',
                                     add_help=True)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--input',
                          help='Path to .tif or .lif input file.',
                          required=True)
    required.add_argument('--outputs',
                          help='Paths to save outputs to. Filenames should '
                               'end with .tif or .ims. {name} is a placeholder'
                               'for series name.',
                          required=True,
                          nargs='+')
    required.add_argument('--model',
                          help='Path to .pkl model (expects also the '
                               'existence of corresponding yaml files).',
                          required=True)

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--channels',
                          default=None,
                          help='Channels with optional preprocessings to use '
                               'e.g. "1,2" or "1-equal,1".')
    optional.add_argument('--labels',
                          default=None,
                          help='Path to instance segmentation labels. It '
                               'should be relative path starting at '
                               '`output_dir`.')
    optional.add_argument('--series',
                          type=int,
                          default=None,
                          help='Which series to segment (.lif files).')
    optional.add_argument('--discrete',
                          choices=['binary', '4bins'],
                          default=None,
                          help='What kind of discretization to use. If not '
                               'given raw probabilities are returned.')
    optional.add_argument('--channel_color',
                          default='Blue',
                          help='Color of input and output channel in .ims '
                               'file.')
    optional.add_argument('--channel_name',
                          default='Classes',
                          help='Name of output channel in .ims file.')
    optional.add_argument('--start',
                          type=int,
                          default=0,
                          help='Starting slice for segmentation.')
    optional.add_argument('--stop',
                          type=int,
                          default=None,
                          help='Last slice for segmentation. It is not '
                               'segmented. If not given, all '
                               'slices from the starting slice are segmented.')
    optional.add_argument('--uff_colors',
                          nargs=2,
                          default=['red', 'green'],
                          help='Colors for negative and positive classes in '
                               '.uff file.')

    model_parameters = parser.add_argument_group('model bound arguments',
                                                 description='Parameters included here should not be manually passed '
                                                             'as they have to comply with configuration used in '
                                                             'training. Normally they are loaded from corresponding '
                                                             'yaml file.')
    model_parameters.add_argument('--use_cubes',
                                  default=None,
                                  type=float,
                                  help='If provided cubes of the given size in um are cropped and passed to '
                                       'classificator')
    model_parameters.add_argument('--features_type',
                                  choices=['default'],
                                  default=None,
                                  help='Identifier of features to extract.')
    model_parameters.add_argument('--instance_model',
                                  default=None,
                                  help='Path to .h5 instance segmentation model ('
                                       'expects also the existence of corresponding '
                                       'yaml files).')
    return parser


def classify(input,
             outputs,
             model,
             features_type=None,
             channels=None,
             labels=None,
             instance_model=None,
             series=0,
             discrete=None,
             channel_color='Blue',
             channel_name='Classes',
             start=0,
             stop=None,
             uff_colors=('red', 'green'),
             use_cubes=None,
             voxel_resize=True):
    """Classify one series.

    See parser for information about arguments.
    """
    with readers.get_volume_reader(input, series=series) as volume_iter:
        input_volume = volume_iter[start:stop]

        for path in outputs:
            utils.ensure_path(path, extensions=('.tif', '.ims'))

        print('Ensuring labels exist...')
        ensure_labels(input=input,
                      output=labels,
                      model=instance_model,
                      channel=0,
                      series=series,
                      start=start,
                      stop=stop)

        with readers.get_volume_reader(labels) as labels_volume:
            input_voxel_size = input_volume.voxel_size
            if not voxel_resize:
                input_voxel_size = None
            elif input_voxel_size is None:
                print("Voxel resize expected but input data has no voxel size.")

            features = None
            if features_type is not None:
                print('Extracting features...')
                features = get_features(input_volume,
                                        labels_volume,
                                        channels,
                                        features_type,
                                        input_voxel_size)

            cropped_cubes = None
            if use_cubes is not None:
                print('Extracting cell cubes...')
                cropped_cubes = get_cell_crops(input_volume, labels_volume, channels, use_cubes,
                                               voxel_size=input_voxel_size)

            print('Overlaying classes on cells...')

            overlayed = overlay_predictions(model, labels_volume.to_numpy(), features, cropped_cubes, discrete)

            print('Saving results...')

            for path in outputs:
                if path.endswith('.tif'):
                    imageio.mimwrite(path, overlayed)
                elif path.endswith('.ims'):
                    channels_preprocess_list = ex.parse_channels_preprocessing(
                        channels)
                    channels = ex.extract_channels(channels_preprocess_list)
                    write_imaris(path,
                                 input_volume[..., 0],
                                 labels_volume,
                                 input_volume,
                                 channels,
                                 overlayed,
                                 channel_name,
                                 channel_color,
                                 metadata=volume_iter.metadata)
                else:
                    write_uff(path, metadata=volume_iter.metadata, classes=overlayed,
                              classif_name=channel_name, colors=uff_colors)


def main(arguments):
    """Run classification on given input.

    See parser for information about arguments.
    """
    yaml_path = yaml_utils.yaml_file(arguments.model)
    args = yutils.merge_yaml_to_cli(
        yaml_path, vars(arguments),
        ['features_type', 'channels', 'instance_model', 'no_voxel_resize', 'use_cubes'])
    yaml_channels = ex.extract_channels(args.channels)
    if arguments.channels is not None:
        cli_channels = ex.extract_channels(arguments.channels)
        translate_table = {ord(str(yaml)): str(cli)
                           for yaml, cli in zip(yaml_channels, cli_channels)}
        final_channels = args.channels.translate(translate_table)
    else:
        final_channels = args.channels

    yutils.update_args(args, channels=final_channels,
                       instance_model=arguments.instance_model)

    if args.series is not None and not args.input.endswith('.lif'):
        warnings.warn(
            'Used series argument with .tif or .uff file, it will be ignored.')
        args.series = 0

    metadata = readers.get_metadata(args.input, series=args.series)
    name = metadata.get('Name', 'series_{}'.format(args.series))
    args.outputs = [path.format(name=name) for path in args['outputs']]
    if args.labels is None:
        args['labels'] = args['outputs'][0].format(name=name) + ".segment.tif"
    else:
        args['labels'] = args.labels.format(name=name)

    args['voxel_resize'] = not args.get('no_voxel_resize', True)  # if model is old then there was no resize
    if 'no_voxel_resize' in args:
        del args['no_voxel_resize']

    for path in args.outputs:
        utils.ensure_path(path, extensions=('.tif', '.ims'))
        output_dir, basename = os.path.split(path)
        basename = os.path.splitext(basename)[0]

        args_path = os.path.join(output_dir, basename + '_args')
        yutils.save_args_to_yaml(args_path, args)

    classify(**args)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
