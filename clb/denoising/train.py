"""Module defines tools for training denoising model.

There are functions that are training model as well as functions that
aggregate operations from `io` and `preprocess` modules to make them easier to
use in main training function.
"""
import os
import pathlib

import fire
import keras.callbacks as callbacks
import keras.optimizers as optimizers

import clb.dataprep.utils as utils
import clb.denoising.architecture as architecture
import clb.denoising.io as denoiseio
import clb.denoising.preprocess as preproc
import clb.denoising.tb_callbacks as tb_callbacks


def prepare_directory_tree(root_dir):
    """Create directory tree for training results.

    Directory tree looks like this (in parentheses on right are names of keys
    in dictionary returned by this function):
    root_dir/                     (root_dir)
             args                 (args_file)
             logs.csv             (csv_logs)
             checkpoints/         (checkpoints_dir)
             tb_logs/             (tb_dir)
                     train/       (tb_train_dir)
                     val/         (tb_val_dir)

    Args:
        root_dir (str): Main directory where results will be saved.

    Returns:
        dict: Dictionary with paths as values.
    """
    checkpoints_dir = os.path.join(root_dir, 'checkpoints')
    os.makedirs(checkpoints_dir)

    tb_dir = os.path.join(root_dir, 'tb_logs')
    tb_train_dir = os.path.join(tb_dir, 'train')
    os.makedirs(tb_train_dir)

    tb_val_dir = os.path.join(tb_dir, 'val')
    os.makedirs(tb_val_dir)

    dirs = locals()
    dirs['args_file'] = os.path.join(root_dir, 'args')
    dirs['csv_logs'] = os.path.join(root_dir, 'logs.csv')

    return dirs


def make_training_pipeline(fovs_pattern, batch_size, shuffle=True, seed=3, augment=True,
                           infinite=False):
    """Create dataset for training.

    Args:
        fovs_pattern (str): Pattern with training fovs.
        batch_size (int): Size of one batch. Last one may be smaller if there is a
                          remainder.
        shuffle (bool): Should paths be shuffled.
        seed (int): Seed for shuffling randomness.
        augment (bool): Should data be augmented.
        infinite (bool): Should data be yielded infinitely.

    Returns:
        preproc.Dataset: Dataset with all specified preprocessings applied.
    """
    dataset = preproc.Dataset(denoiseio.list_fovs(fovs_pattern, shuffle=shuffle,
                                                  seed=seed),
                              infinite=infinite)

    dataset = dataset.transform(denoiseio.read_png)

    if augment:
        dataset = dataset.transform(preproc.augment)

    dataset = dataset.batch(batch_size=batch_size).transform(utils.ensure_4d)

    return dataset


def main(train_fovs_pattern, val_fovs_pattern, batch_size, epochs, learning_rate,
         save_dir='./logs', augment=True, shuffle=True, seed=3, model_save_frequency=1):
    """Train the network.

    Args:
        train_fovs_pattern (str): Pattern for training fovs.
        val_fovs_pattern (str): Pattern for validation fovs.
        batch_size (int): Size of a mini-batch.
        epochs (int): Number of epochs.
        learning_rate (float):
        save_dir (str): Root path for stuff saved during training.
        augment (bool): Should data be augmented.
        shuffle (bool): Should data be shuffled.
        seed (int): Seed for shuffling randomness.
        model_save_frequency (int): Model save frequency. Model will be saved every
                                    `model_save_frequency` epochs and also when
                                    validation loss improves.
    """
    train_dataset = make_training_pipeline(fovs_pattern=train_fovs_pattern,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           seed=seed,
                                           augment=augment,
                                           infinite=True)
    val_dataset = make_training_pipeline(fovs_pattern=val_fovs_pattern,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         seed=seed,
                                         augment=augment,
                                         infinite=True)

    optimizer = optimizers.Adam(lr=learning_rate)
    model = architecture.build_model(optimizer=optimizer,
                                     loss='mse')

    dirs = prepare_directory_tree(save_dir)

    cbs = [
        callbacks.TensorBoard(log_dir=dirs['tb_dir']),
        callbacks.CSVLogger(filename=dirs['csv_logs']),
        tb_callbacks.TensorBoardImage(log_dir=dirs['tb_dir'], dataset=val_dataset),
        callbacks.ModelCheckpoint(filepath=str(pathlib.Path(dirs['checkpoints_dir'])
                                               / 'best-val-loss.h5'),
                                  monitor='val_loss',
                                  mode='min',
                                  save_best_only=True,
                                  verbose=1),
        callbacks.ModelCheckpoint(filepath=str(pathlib.Path(dirs['checkpoints_dir'])
                                               / 'checkpoint.h5'),
                                  period=model_save_frequency)
    ]
    model.fit_generator(train_dataset,
                        epochs=epochs,
                        validation_data=val_dataset,
                        validation_steps=len(val_dataset),
                        steps_per_epoch=len(train_dataset),
                        callbacks=cbs,
                        use_multiprocessing=True)


if __name__ == '__main__':
    fire.Fire(main)
