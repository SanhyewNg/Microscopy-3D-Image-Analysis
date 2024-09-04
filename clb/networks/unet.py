"""
This is a U-net written for the segmentation purposes but due to the sudden
project cancelation hasn't been trained on the available data. Previous
Bartek Miselis' work suggest that it would get better results than DCAN so it is
definitely worth checking out. It was tested empirically, the architecture
graph was checked and correct. It is ready to run and train.

"""

from keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D,
                          concatenate, UpSampling2D, Activation)
from keras.models import Model

from clb.networks.utils import LayersFiltersIterator
option_dict_conv = {"activation": "relu", "padding": "same"}
option_dict_bn = {"momentum": 0.9}


def build_unet(dim1, dim2, input_channels=1, output_channels=3, unet_levels=4,
               start_filters=64, activation="softmax"):
    """Builds a unet model based on the input params.

    Args:
        dim1 (int): Length of the input image first dimension.
        dim2 (int): Length of the input image second dimension.
        input_channels (int):  Number of input's channels.
        output_channels (int): Number of output's channels.
        unet_levels (int): Number of levels in the created U-net model.
        start_filters (int): Number of filters ath the shallowest level of U-net
        activation (str): Name of the activation function
    """
    # encoding part
    inp = Input(shape=(dim1, dim2, input_channels))

    enc = inp
    enc_list = []

    for _, filters in LayersFiltersIterator(unet_levels, start_filters,
                                            going_down=True):
        for i in range(2):
            enc = Conv2D(filters, 3, **option_dict_conv)(enc)
            enc = BatchNormalization(**option_dict_bn)(enc)
        enc_list.append(enc)
        enc = MaxPooling2D()(enc)

    # decoding part
    dec = enc
    for layer_index, filters in LayersFiltersIterator(unet_levels,
                                                      start_filters,
                                                      going_down=False):
        dec = UpSampling2D()(dec)
        dec = concatenate([dec, enc_list[layer_index - 1]], axis=3)
        for i in range(2):
            dec = Conv2D(256, 3, **option_dict_conv)(dec)
            dec = BatchNormalization(**option_dict_bn)(dec)

    out = Conv2D(output_channels, 3, **option_dict_conv)(dec)
    out = Activation(activation)(out)
    model = Model(inp, out)

    return model


if __name__ == '__main__':
    build_unet(256, 256, 3)
