"""hypertune - automatically find best learning rate and batch size.

Whole idea is inspired by http://cs231n.github.io/neural-networks-3/#hyper.
"""
import os

import fire
import numpy as np
import tensorflow as tf

from clb.train.train import train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Hypertune:
    def tune_lr(self, architecture, batch, train_data, val_data, seed=48):
        """Tune learning rate.

        Args:
            architecture (str): network architecture either 'unet' or 'dcan'.
            batch (int): Batch size.
            train_data (str): Path to directory with the data that will be
                              used as a train dataset. Multiple dirs ar
                              allowed, just split them with a '+' character.
            val_data: (str): Path to directory with the data that will be
                             used as a validation dataset. Multiple dirs are
                             allowed, just split them with a + character.
            seed (int): Seed all randomness.

        Outputs:
            Best power, best learning rate and best validation loss to file
            models/best_final_pow.txt.
        """
        np.random.seed(seed)
        tf.set_random_seed(seed)
        pow_phase_1, _ = find_best_power(stage=1, min_pow=-6, max_pow=-1,
                                         num_lrs=20, epochs=5, augs=10,
                                         seed=seed, batch=batch,
                                         trim_method='resize',
                                         train_data=train_data,
                                         val_data=val_data)

        pow_phase_2, _ = find_best_power(stage=2, min_pow=pow_phase_1 - 1,
                                         max_pow=pow_phase_1 + 1, num_lrs=10,
                                         epochs=25, augs=10, seed=seed,
                                         batch=batch,
                                         trim_method='resize',
                                         train_data=train_data,
                                         val_data=val_data)

        best_pow, best_val_loss = find_best_power(stage=3,
                                                  min_pow=pow_phase_2 - 0.1,
                                                  max_pow=pow_phase_1 + 0.1,
                                                  num_lrs=5, epochs=50,
                                                  augs=10,
                                                  seed=seed,
                                                  batch=batch,
                                                  trim_method='resize',
                                                  train_data=train_data,
                                                  val_data=val_data)

        with open('models/best_final_pow.txt', 'w') as f:
            f.write('best pow: {}\n'.format(best_pow))
            f.write('best lr: {0:.9f}'.format(10 ** best_pow))
            f.write('best val_loss: {}'.format(best_val_loss))

    def tune_batch(self, architecture, epochs, train_data, val_data, augs=0,
                   lr=0.001, seed=48):
        """Tune batch size.

        Args:
            architecture (str): network architecture either 'unet' or 'dcan'.
            epochs (int): Number of epochs.
            train_data (str): Path to directory with the data that will be
                              used as a train dataset. Multiple dirs are
                              allowed, just split them with a + character.
            val_data (str): Path to directory with the data that will be
                            used as a train dataset. Multiple dirs are
                            allowed, just split them with a + character.
            augs (int): Number of augmentations.
            lr (float): Learning rate.
            seed (int): Seed all

        Outputs:
            Best batch size, best validation loss to file
            models/best_final_batch.txt.
        """
        np.random.seed(seed)
        tf.set_random_seed(seed)
        batch_sizes = list(2 ** np.array(range(1, 6)))

        batch_dict = {}
        for batch_size in batch_sizes:
            val_loss = train(architecture=architecture,
                             epochs=epochs,
                             batch=batch_size,
                             augments=augs,
                             lr=lr,
                             model='models/model_batch_'
                                   '{}_lr_{:.9f}.h5'.format(batch_size, lr),
                             train_data=train_data,
                             trim_method='resize',
                             val_data=val_data,
                             seed=seed)

            print(
                '\n\nBATCH: {}\nVAL_LOSS: {}\n\n'.format(batch_size, val_loss))
            batch_dict[batch_size] = val_loss

        best_batch = min(batch_dict.items(), key=lambda x: x[1])[0]

        # Clean models that are not the best.
        for batch_size in batch_sizes:
            if batch_size != best_batch:
                os.remove('models/model_batch_{}_lr_{:.9f}.h5'.format(
                    batch_size, lr))
                os.remove('models/model_batch_{}_lr_{:.9f}.yaml'.format(
                    batch_size, lr))

        with open('models/best_final_batch.txt', 'w') as f:
            f.write('best batch: {}\n'.format(best_batch))
            f.write('best val_loss: {}'.format(batch_dict[best_batch]))


def find_best_power(stage, min_pow, max_pow, num_lrs, architecture, epochs,
                    augs, trim_method, seed, batch, train_data, val_data):
    """Find best power (in mathematic sense) for learning rate.

    In hypertune script the assumption is that:
    learning rate = 10 ** (different powers)

    This function is used to run training across different values of
    learning rate (by modifying it's power, not learning rate itself). Why
    power and not learning rate directly? Mainly because it's easier to
    perform staged search (see 'Stage your search from course to fine' from
    http://cs231n.github.io/neural-networks-3/#hyper).

    Best power is chosen based on validation loss and thus may not fully
    satisfy your case, when there're multiple possible trade-offs between
    various metrics.

    Args:
        stage: which searching stage is being run (just for naming models
               and printing info)
        min_pow: minimum power value in current stage
        max_pow: maximum power value in current stage
        num_lrs: number of different learning rates to try
        architecture (str): network architecture either 'unet' or 'dcan'.
        epochs: number of epochs to run single training
        augs: number of image augmentations in single training
        trim_method: trimming method used for single training ('padding',
                     'reflect' or 'resize')
        seed: value for random number generator's seed
        batch: size of batch in single training
        train_data: training data used to train the network
        val_data: validation data uset to train the network

    Returns:
        best power, along with best validation loss.
    """
    powers = np.random.uniform(min_pow, max_pow, num_lrs)
    powers = [round(pow, 9) for pow in powers]

    pow_dict = {}
    for pow in powers:
        lr = 10 ** pow
        val_loss = train(architecture=architecture,
                         epochs=epochs,
                         batch=batch,
                         augments=augs,
                         trim_method=trim_method, lr=lr,
                         train_data=train_data, val_data=val_data,
                         model='models/model_phase_{}_lr_{:.9f}.h5'.format(
                           stage, lr), seed=seed)

        pow_dict[pow] = val_loss
        print('\n\nLR: {:.9f}\nVAL_LOSS: {}\n\n'.format(lr, val_loss))

    best_pow = min(pow_dict.items(), key=lambda x: x[1])[0]
    print('\n\n@@@ STAGE: {}\n    LR: {:.9f}\n    VAL_LOSS: {}\n\n'.format(
        stage,
        10 ** best_pow,
        pow_dict[best_pow]))

    # Clean models that are not the best.
    for pow in powers:
        lr = 10 ** pow

        if pow != best_pow:
            os.remove('models/model_phase_{}_lr_{:.9f}.h5'.format(stage, lr))
            os.remove('models/model_phase_{}_lr_{:.9f}.yaml'.format(stage, lr))

    best_val_loss = pow_dict[best_pow]
    return best_pow, best_val_loss


if __name__ == '__main__':
    fire.Fire(Hypertune)
