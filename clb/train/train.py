import os
from itertools import cycle

import fire
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)
from tqdm import tqdm

import clb.train.tbcallbacks as tbc
from clb.dataprep.generators import (dcan_dataset_generator,
                                     unet_dataset_generator)
from clb.networks import dcan, unet
from clb.train.losses import weighted_crossentropy
from clb.train.metrics import (channel_iou, channel_precision, channel_recall,
                               iou)
from clb.train.utils import count_steps, get_images_for_tensorboard
from clb.yaml_utils import load_args, save_args_to_yaml


@load_args(arg_name='args_load_path')
def train(*, architecture, epochs, batch, augments, trim_method,
          train_data, val_data, model, channels=1, lr=1e-3, im_dim=256,
          bnd_weight=0.5, obj_weight=0.5, boundary_thickness=2, enable_elastic=True,
          remove_blobs=False, touching_boundaries=False, dropout=0.5,
          dcan_final_act='tanh', optimizer='optimizers.Adam(lr=lr)',
          unet_levels=4, start_filters=64, tb_dir='tensorboard/',
          csv_log='log.csv', seed=48, verbose=2, in_memory=False,
          preprocessings=()):
    """
    Train model.

    Args:
        architecture (str): either 'dcan' or 'unet' for now.
        epochs (int): Number of epochs.
        batch (int): Number of images in a batch.
        augments (int): Number of augumentations per sample.
        trim_method (str): Method of adapting input size to expected by network
                           (resize, padding, reflect).
        train_data (str): Path to directory with the data that will be used
                          as a train dataset. Multiple dirs are allowed,
                          just split them with a '+' character.
        val_data (str): Path to directory with the data that will be used as a
                        validation dataset. Multiple dirs are allowed,
                        just split them with a '+' character.
        model (str): Specify path (with name) to save the model.
        channels (int): Specify the number of input's channels. At this moment,
                        it defines how many images will be used for the third
                        dimensional spatial context in the network.
        lr (float): Learning rate.
        im_dim (int): Size of the image entering the network (network
                      receptive field).
        bnd_weight (float): Loss weight for boundaries loss.
        obj_weight (float): Loss weight for object loss.
        boundary_thickness (int): how thick is the boundary (by default it's
                                  2 pixels thick). It should always be an
                                  even number.
        enable_elastic (bool): should elastic distortions be applied as part
                               of augmentations.
        remove_blobs (bool): if True, removing blobs.
        touching_boundaries (bool): if True results in binary masks only with
                                    boundaries that are touching.
        dropout (float): Dropout probability.
        dcan_final_act (str): Final activation of DCAN model.
        optimizer (str): string representing optimizer that will be used for
                         training. This expression is evaluated.
        tb_dir (str): Output for tensorboard logs.
        csv_log (str): Path to save CSV log from training.
        seed (int): Set seed for all random generators.
        verbose (int): value [0, 1, 2] to set training verbosity mode.
        in_memory (int): if non-zero then the generators are used to prepare static
                         train and val datasets before fitting.
        preprocessings (iterable): Preprocessings to use, currently supported:
                                   - denoising
                                   - clahe

    Returns:
        Model training history.
    """
    if architecture not in ['dcan', 'unet']:
        raise ValueError("--architecture must be either 'dcan' or 'unet'.")
    if architecture == "dcan" and channels != 1:
        raise ValueError("Multiple channels aren't supported by DCAN")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    save_args_to_yaml(os.path.splitext(model)[0], locals())

    try:
        tb_dir = os.path.join(os.environ['AZ_BATCHAI_OUTPUT_OUT'], tb_dir)
        model = os.path.join(os.environ['AZ_BATCHAI_OUTPUT_OUT'], model)
        csv_log = os.path.join(os.environ['AZ_BATCHAI_OUTPUT_OUT'], csv_log)
    except KeyError:
        pass

    train_dirs = train_data.split('+')
    val_dirs = val_data.split('+')

    train_img_data = [os.path.join(dir, 'images') for dir in train_dirs]
    train_gt_data = [os.path.join(dir, 'labels') for dir in train_dirs]
    val_img_data = [os.path.join(dir, 'images') for dir in val_dirs]
    val_gt_data = [os.path.join(dir, 'labels') for dir in val_dirs]

    if architecture == 'dcan':
        train_gen = dcan_dataset_generator(image_data=train_img_data,
                                           gt_data=train_gt_data,
                                           batch_size=batch,
                                           out_dim=im_dim,
                                           trim_method=trim_method,
                                           augs=augments,
                                           seed=seed,
                                           boundary_thickness=boundary_thickness,
                                           enable_elastic=enable_elastic,
                                           remove_blobs=remove_blobs,
                                           touching_boundaries=touching_boundaries,
                                           infinite=True,
                                           preprocessings=preprocessings)

        val_gen = dcan_dataset_generator(image_data=val_img_data,
                                         gt_data=val_gt_data,
                                         batch_size=batch,
                                         out_dim=im_dim,
                                         trim_method=trim_method,
                                         augs=0,
                                         seed=seed,
                                         boundary_thickness=boundary_thickness,
                                         enable_elastic=enable_elastic,
                                         remove_blobs=remove_blobs,
                                         touching_boundaries=touching_boundaries,
                                         infinite=True,
                                         preprocessings=preprocessings)
    elif architecture == 'unet':
        train_gen = unet_dataset_generator(image_data=train_img_data,
                                           gt_data=train_gt_data,
                                           channels=channels,
                                           batch_size=batch,
                                           out_dim=im_dim,
                                           trim_method=trim_method,
                                           augs=augments,
                                           seed=seed,
                                           boundary_thickness=boundary_thickness,
                                           enable_elastic=enable_elastic,
                                           remove_blobs=remove_blobs,
                                           touching_boundaries=touching_boundaries,
                                           infinite=True)

        val_gen = unet_dataset_generator(image_data=val_img_data,
                                         gt_data=val_gt_data,
                                         channels=channels,
                                         batch_size=batch,
                                         out_dim=im_dim,
                                         trim_method=trim_method,
                                         augs=0,
                                         seed=seed,
                                         boundary_thickness=boundary_thickness,
                                         enable_elastic=enable_elastic,
                                         remove_blobs=remove_blobs,
                                         touching_boundaries=touching_boundaries,
                                         infinite=True)

    # Build the model
    input_shape = (im_dim, im_dim, channels)

    if architecture == 'dcan':
        network_model = dcan.build(input_shape=input_shape,
                                   final_activation=dcan_final_act,
                                   dropout=dropout)

        network_model.compile(optimizer=eval(optimizer),
                              loss={'objects': 'binary_crossentropy',
                                    'boundaries': 'binary_crossentropy'},
                              loss_weights={'objects': obj_weight,
                                            'boundaries': bnd_weight},
                              metrics=[iou])
    elif architecture == 'unet':
        network_model = unet.build_unet(dim1=im_dim,
                                        dim2=im_dim,
                                        channels=channels,
                                        unet_levels=4,
                                        start_filters=64)

        metrics = [
           channel_recall(channel=0, name="background_recall"),
           channel_precision(channel=0, name="background_precision"),
           channel_recall(channel=1, name="objects_recall"),
           channel_precision(channel=1, name="objects_precision"),
           channel_recall(channel=2, name="boundaries_recall"),
           channel_precision(channel=2, name="boundaries_precision"),
           channel_iou(channel=0, name="background_iou"),
           channel_iou(channel=1, name="objects_iou"),
           channel_iou(channel=2, name="boundaries_iou"),
          ]

        network_model.compile(optimizer=eval(optimizer),
                              loss=weighted_crossentropy,
                              metrics=metrics)

    # Prepare subset of validation images to be previewed during training
    # in TensorBoard.
    #val_gen, val_gen_copy = tee(val_gen)
    display_train_x, display_train_y = get_images_for_tensorboard(batch_gen=train_gen,
                                                      num_imgs=5,
                                                      architecture=architecture)

    display_x, display_y = get_images_for_tensorboard(batch_gen=val_gen,
                                                      num_imgs=5,
                                                      architecture=architecture)

    # Callbacks.
    cbs = []
    cbs.append(tbc.TensorBoard(log_dir=tb_dir))
    cbs.append(CSVLogger(filename=csv_log))
    cbs.append(ReduceLROnPlateau(monitor='val_loss', factor=1e-1,
                                 patience=8, min_lr=1e-12, verbose=1))
    cbs.append(EarlyStopping(monitor='val_loss',
                             mode='min',
                             min_delta=1e-4,
                             patience=15,
                             verbose=1))

    clean_model_name, _ = os.path.splitext(model)
    cbs.append(ModelCheckpoint(filepath=clean_model_name +
                               '-best-val-loss.h5',
                               monitor='val_loss',
                               mode='min',
                               save_best_only=True,
                               verbose=1))
    cbs.append(ModelCheckpoint(filepath=clean_model_name +
                               '-best-val-objects-iou.h5',
                               monitor='val_objects_iou',
                               mode='max',
                               save_best_only=True,
                               verbose=1))
    cbs.append(ModelCheckpoint(filepath=clean_model_name +
                               '-best-val-boundaries-iou.h5',
                               monitor='val_boundaries_iou',
                               mode='max',
                               save_best_only=True,
                               verbose=1))
    cbs.append(tbc.Segmentation(Xs=display_x,
                                Ys=display_y,
                                log_dir=os.path.join(tb_dir, 'predictions'),
                                architecture=architecture))

    cbs.append(tbc.Segmentation(Xs=display_train_x,
                                Ys=display_train_y,
                                log_dir=os.path.join(tb_dir, 'predictions_train'),
                                architecture=architecture))

    n_train = count_steps(train_img_data, channels)
    n_val = count_steps(val_img_data, channels)

    steps_per_epoch = (n_train + augments * n_train) // batch + 1
    val_steps = (n_val + augments * n_val) // batch + 1

    if in_memory:
        train_gen_first = [x for _, x in tqdm(zip(range(steps_per_epoch * in_memory), train_gen),
                                              desc="Generating train dataset", total=steps_per_epoch * in_memory)]
        train_gen = cycle(train_gen_first)

        val_gen_first = [x for _, x in tqdm(zip(range(val_steps), val_gen),
                                            desc="Generating validation dataset", total=val_steps)]
        val_gen = cycle(val_gen_first)

    history = network_model.fit_generator(generator=train_gen,
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=epochs,
                                          verbose=verbose,
                                          callbacks=cbs,
                                          validation_data=val_gen,
                                          validation_steps=val_steps,
                                          workers=0,
                                          use_multiprocessing=False)

    return min(history.history['val_loss'])


if __name__ == '__main__':
    fire.Fire(train)
