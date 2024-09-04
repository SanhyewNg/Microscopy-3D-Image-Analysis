import os
from itertools import tee

from attrdict import AttrDict
import fire
import numpy as np
from keras.preprocessing.image import array_to_img

from clb.predict.predict_tile import (get_probability_calculator,
                                      load_model_with_cache)
from clb.train.utils import count_steps, plot_pred_results
from clb.dataprep.generators import (dcan_dataset_generator,
                                     unet_dataset_generator)
from clb.yaml_utils import merge_cli_to_yaml
from clb.segment.instance.components import connected_components


def predict(config, test_data, output=None, plot=False):
    """
    Make predictions.

    Args:
        config (str): Path to configuration file generated during training.
        test_data (str): Path to directory with the data that will be used
                         as a test dataset. Multiple dirs are allowed,
                         just split them with a + character.
        output (str): Path to segmentation results.
        plot (bool): Enable plotting network results.

    Returns:
        If `output` is None, predictions are not saved on the disk. If `output`
        is a valid path, predictions are saved on the disk.
    """
    # Get function parameters into cli_args.
    cli_args = locals()

    # Load config from yaml file and merge it with CLI arguments.
    if cli_args['config'] is not None:
        args = AttrDict(merge_cli_to_yaml(cli_args['config'], cli_args,
                                          params_to_merge=None))
    else:
        args = AttrDict(cli_args)

    # Check whether model has its architecture specified
    assert args.get('architecture') is not None, ("Please specify model's "
                                                  "architecture in its yaml "
                                                  "file.")
    assert args.get('channels') is not None, ("Please specify number of input's"
                                              " channels in its yaml file")

    if args.test_data is not None:
        test_dirs = args.test_data.split('+')

        test_img_data = [os.path.join(dir, 'images') for dir in test_dirs]
        test_gt_data = [os.path.join(dir, 'labels') for dir in test_dirs]

        if args.architecture == 'dcan':
            test_gen, test_gen_vis = tee(dcan_dataset_generator
                                         (image_data=test_img_data,
                                          gt_data=test_gt_data,
                                          batch_size=1,
                                          out_dim=args.im_dim,
                                          trim_method=args.trim_method,
                                          augs=0,
                                          seed=args.seed,
                                          infinite=False))
        elif args.architecture == 'unet':
            test_gen, test_gen_vis = tee(unet_dataset_generator
                                         (image_data=test_img_data,
                                          gt_data=test_gt_data,
                                          channels=args.channels,
                                          batch_size=1,
                                          out_dim=args.im_dim,
                                          trim_method=args.trim_method,
                                          augs=0,
                                          seed=args.seed,
                                          enable_elastic=False,
                                          infinite=False))

    else:
        raise ValueError("No data to predict on! --test_data missing!")

    model = load_model_with_cache(args.model)
    n_test = count_steps(test_img_data, args.channels)
    if args.architecture == 'dcan':

        pred_objects, pred_boundaries = \
            model.predict_generator(generator=test_gen,
                                    steps=n_test,
                                    workers=1,
                                    use_multiprocessing=False,
                                    verbose=1)

    elif args.architecture == 'unet':

        prediction = \
            model.predict_generator(generator=test_gen,
                                    steps=n_test,
                                    workers=1,
                                    use_multiprocessing=False,
                                    verbose=1)

        pred_objects = prediction[:, :, :, 1]
        pred_boundaries = prediction[:, :, :, 2]

    prob_calculator = get_probability_calculator(args.model)
    pred_objects = prob_calculator(pred_objects)
    pred_boundaries = prob_calculator(pred_boundaries)

    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    # Postprocessing to perform basic instance segmentation.
    for idx, ((img, gt), objects, boundaries) in \
            enumerate(zip(test_gen_vis,
                          pred_objects,
                          pred_boundaries)):

        # Remove dimensions of size 1.
        objects = np.squeeze(objects)
        boundaries = np.squeeze(boundaries)

        # Preview outputs.
        if args.plot:
            plot_pred_results(args.architecture, img, gt, objects, boundaries)

        # Perform postprocessing.
        original, postprocessed_labels = connected_components(img,
                                                              objects,
                                                              boundaries,
                                                              args.plot)

        # Save predictions as png files.
        if args.output:
            img = array_to_img(postprocessed_labels)
            img.save(os.path.join(args.output, 'prediction_{}.png'.format(idx)))

            orig_img = array_to_img(np.squeeze(original))
            orig_img.save(os.path.join(args.output, 'orig_{}.png'.format(idx)))


if __name__ == '__main__':
    fire.Fire(predict)
