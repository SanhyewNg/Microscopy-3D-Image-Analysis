"""Module with network architecture.

Attributes:
    ENCODER_FILTER_NUMBERS (tuple): Lists with numbers of filters in each encoder block.
    DECODER_FILTER_NUMBERS (tuple): Lists with numbers of filters in each decoder block.
    FINAL_ACTIVATION (None|str): Activation on the final layer.
"""
import keras.layers as layers
import keras.models as models
import numpy as np


ENCODER_FILTER_NUMBERS = (
    [48, 48],
    [48],
    [48],
    [48],
    [48],
    [48]
)

DECODER_FILTER_NUMBERS = (
    [96, 96],
    [96, 96],
    [96, 96],
    [96, 96],
    [64, 32]
)

FINAL_ACTIVATION = 'tanh'


def add_downsample_block(input_layer, filter_numbers, skip_stack, pooling=True):
    """Add couple of convolutions followed by an optional max pooling.

    If there is pooling used last convolution is pushed on `skip_stack` to make it
    possible to add skip connection.

    Args:
        input_layer (keras.Layer): Input layer to the block.
        filter_numbers (Iterable): Numbers of filters for each convolution.
        skip_stack (list): Stack for easier creation of skip connections.
        pooling (bool): Should max pooling be used at the end of the block.

    Returns:
        keras.Layer: Output layer of the block.
    """
    current_layer = input_layer
    for filter_number in filter_numbers[:-1]:
        conv_layer = layers.Conv2D(filters=filter_number,
                                   padding='same',
                                   kernel_size=(3, 3))(current_layer)
        current_layer = layers.LeakyReLU()(conv_layer)

    conv_layer = layers.Conv2D(filters=filter_numbers[-1],
                               padding='same',
                               kernel_size=(3, 3))(current_layer)
    output_layer = layers.LeakyReLU()(conv_layer)
    if pooling:
        skip_stack.append(output_layer)
        output_layer = layers.MaxPool2D(pool_size=(2, 2), padding='same')(output_layer)

    return output_layer


def add_upsample_block(input_layer, filter_numbers, skip_stack):
    """Add upsampling followed by skip connection and couple of convolutions.

    Args:
        input_layer (keras.Layer): Input layer for the block.
        filter_numbers (Iterable): Numbers of filters for each convolution.
        skip_stack (list): Stack with layers for skip connections.

    Returns:
        keras.Layer: Output layer for the block.
    """
    upsample_layer = layers.UpSampling2D(size=(2, 2),
                                         interpolation='nearest')(input_layer)
    skip_connection = layers.Concatenate()([skip_stack.pop(), upsample_layer])
    current_layer = skip_connection
    for filter_number in filter_numbers:
        current_layer = layers.Conv2D(filters=filter_number,
                                      kernel_size=(3, 3),
                                      padding='same')(current_layer)
        current_layer = layers.LeakyReLU()(current_layer)

    return current_layer


def build_model(optimizer, loss, encoder_filter_numbers=ENCODER_FILTER_NUMBERS,
                decoder_filter_numbers=DECODER_FILTER_NUMBERS):
    """Build unet.

    Args:
        optimizer (str|tf.keras.Optimizer): Optimizer used in network training.
        loss (str|tf.keras.Loss): Loss used in network training.
        encoder_filter_numbers (Iterable): Each element of it should be an Iterable with
                                           numbers of filters in convolutions for the
                                           block.
        decoder_filter_numbers (Iterable): Same as `encoder_filter_numbers`, but for the
                                           decoder part.

    Returns:
        keras.Model: Compiled model.
    """
    input_layer = layers.Input(shape=(None, None, 1), dtype=np.float32)
    skip_stack = []

    # Encoder
    current_layer = input_layer
    for filter_numbers in encoder_filter_numbers[:-1]:
        current_layer = add_downsample_block(current_layer, filter_numbers, skip_stack)
    current_layer = add_downsample_block(current_layer, encoder_filter_numbers[-1],
                                         skip_stack, pooling=False)

    # Decoder
    for filter_numbers in decoder_filter_numbers:
        current_layer = add_upsample_block(current_layer, filter_numbers, skip_stack)
    output_layer = layers.Conv2D(filters=1, kernel_size=(3, 3),
                                 padding='same',
                                 activation=FINAL_ACTIVATION)(current_layer)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss=loss)

    return model
