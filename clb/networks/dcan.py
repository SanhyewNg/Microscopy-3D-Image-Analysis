import numpy as np
from keras.layers import (Activation, Add, BatchNormalization, Conv2D,
                          Conv2DTranspose, Dropout, Input, MaxPooling2D)
from keras.models import Model


def get_probability_from_sigmoid(raw_prediction):
    """
    Sigmoid activation function operating on values >=0 from relus so it
    returns values in range [0.5, 1.0].
    Args:
        raw_prediction: numpy array from network with relus+sigmoid

    Returns:
        image with values in range [0.0, 1.0]
    """
    assert ((0.5 <= raw_prediction) & (raw_prediction <= 1.0)).all(), \
        "prediction of sigmoid have unexpected values"
    return 2 * raw_prediction - 1


def get_probability_from_tanh(raw_prediction):
    """
    Tanh activation function returns values in range [0.0, 1.0] so there is
    no need for transformations.

    Args:
        raw_prediction: numpy array from network with relus+tanh

    Returns:
`       image with values in range [0.0, 1.0]
    """
    # It may be possible that due to floating point number rounding max value
    # can be slightly greater than 1.0, e.g. 1.000001.
    max_diff = np.max(raw_prediction - 1.0)
    if 0.0 < max_diff < 1e-3:
        raw_prediction[raw_prediction > 1.0] = 1.0

    assert ((0.0 <= raw_prediction) & (raw_prediction <= 1.0)).all(), \
        "prediction of tanh have unexpected values"
    return raw_prediction


def conv_batch_relu(filters, kernel_size, inputs, init, momentum):
    """Building block consisting of convolution -> batch norm -> relu.
    """
    conv = Conv2D(filters=filters, kernel_size=kernel_size,
                  kernel_initializer=init, padding='same')(inputs)

    batch_norm = BatchNormalization(momentum=momentum)(conv)
    act = Activation('relu')(batch_norm)

    return act


def conv_transp_batch_relu(filters, kernel_size, strides, inputs, init,
                           momentum):
    """Building block consisting of transposed convolution -> batch norm ->
    relu.
    """
    conv_transp = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                  kernel_initializer=init,
                                  strides=strides, padding='same')(inputs)

    batch_norm = BatchNormalization(momentum=momentum)(conv_transp)
    act = Activation('relu')(batch_norm)
    return act


def down_block(filters, inputs, momentum, init='he_uniform',
               kernel_size=(3, 3), with_pool=True, pool_size=(2, 2)):
    """Building block consisting of convolution -> max pool.
    Args:

    Returns:
        namedtuple with conv and pool to allow for compact code and correct
        conv usage in upsampling branches
    """
    conv = conv_batch_relu(filters=filters, kernel_size=kernel_size,
                           inputs=inputs, init=init,
                           momentum=momentum)

    if with_pool:
        pool = MaxPooling2D(pool_size=pool_size)(conv)
        return pool
    else:
        return conv


def up_block(filters, kernel_size, strides, inputs, dropout, init='he_uniform',
             momentum=0.99):
    """Building block consisting of transposed conv -> batch norm -> relu ->
    conv -> relu -> dropout.
    """
    conv_trans = conv_transp_batch_relu(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides, inputs=inputs,
                                        init=init,
                                        momentum=momentum)

    conv = Conv2D(filters=1, kernel_size=(1, 1),
                  kernel_initializer=init, padding='same',
                  activation='relu')(conv_trans)

    drop = Dropout(dropout)(conv)
    return drop


def get_strides_kernel_size(up_input, orig_dim):
    up_input_dim = int(up_input.get_shape()[1])

    stride = int(orig_dim / up_input_dim)
    strides = (stride, stride)

    kernel_dim = stride * 2
    kernel_size = (kernel_dim, kernel_dim)

    return strides, kernel_size


def up_branch(down_path, orig_dim, branch_depth, dropout):
    """Function to generate whole branch.

    Upsampling branch consists of multiple smaller branches. Smaller
    branches are created from the end of downsampling path up to the begin.
    How many of them will be generated depends on `branch_depth` parameter.

    Let's say `down_path` is built from 6 blocks (DCAN default) and
    `branch_depth = 3`. Calling:
    ```
    up_branch(down_path=down_path, orig_dim=orig_dim, branch_depth=3)
    ```
    will produce an upsampling branch consisting of 3 smaller branches:
    first origins from block 6, second from block 5 and third from block 4.

    Args:
        down_path: downsampling path in list format,
                   e.g. [down_block, down_block, down_block, ...]
        orig_dim: original dimension of the image (squares assumed)
        branch_depth: how many smaller branches will from the upsampling
                      branch. This value has to be smaller then length of
                      the downsampling path.
        dropout: value from range [0.0, 1.0] (probability).

    Returns:
        list of smaller branches (`up_block()`) that from upsampling branch
    """
    if branch_depth >= len(down_path):
        raise ValueError('Upsampling branch cannot be deeper or of equal ' +
                         'depth as the downsampling path.')

    branches = []
    for n in range(-1, -branch_depth - 1, -1):
        up_input = down_path[n]

        strides, kernel_size = get_strides_kernel_size(up_input, orig_dim)

        branches.append(up_block(filters=1, kernel_size=kernel_size,
                                 strides=strides, inputs=up_input,
                                 dropout=dropout))

    return branches


def build(input_shape, final_activation, dropout, momentum=0.99):
    """Deep Contour-Aware Network.

    https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S12-01.pdf

    Args:
        input_shape: shape of the input of the data entering the network
                     (without batch size)
        final_activation: what final activation function should be
        dropout: value from range [0.0, 1.0] (probability).
        momentum: value for momentum parameter in batchnorm layers

    Returns:
        [batch_size, height, width, 2], where 1st channel is reserved
        for objects and 2nd channel is reserved for boundaries

    Raise:
        ValueError when `final_activation` is neither sigmoid nor tanh.
    """
    if final_activation not in ['sigmoid', 'tanh']:
        raise ValueError("Final activation has to be either sigmoid or tanh.")

    net_inputs = Input(shape=input_shape)

    # Downsampling path.
    down = []

    down.append(down_block(filters=64, inputs=net_inputs, momentum=momentum))
    down.append(down_block(filters=128, inputs=down[0], momentum=momentum))
    down.append(down_block(filters=256, inputs=down[1], momentum=momentum))
    down.append(down_block(filters=512, inputs=down[2], with_pool=False,
                           momentum=momentum))
    pool4 = MaxPooling2D(pool_size=(2, 2))(down[3])
    down.append(down_block(filters=512, inputs=pool4, with_pool=False,
                           momentum=momentum))
    pool5 = MaxPooling2D(pool_size=(2, 2))(down[4])
    down.append(down_block(filters=1024, inputs=pool5, with_pool=False,
                           momentum=momentum))

    # Upsampling path (both branch for detecting objects and boundaries) look
    # exactly the same. However, during training, they learn different things
    # (because output from each branch is eventually compared with either
    # ground truth with object or with boundaries).
    up_obj = up_branch(down_path=down, orig_dim=input_shape[0], branch_depth=3,
                       dropout=dropout)
    up_bnd = up_branch(down_path=down, orig_dim=input_shape[0], branch_depth=3,
                       dropout=dropout)

    objects = Add()(up_obj)
    objects_act = Activation(final_activation, name='objects')(objects)

    boundaries = Add()(up_bnd)
    boundaries_act = Activation(final_activation, name='boundaries')(boundaries)

    model = Model(inputs=[net_inputs], outputs=[objects_act, boundaries_act])

    return model
