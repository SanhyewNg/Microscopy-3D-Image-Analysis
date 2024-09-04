import os

import keras.layers
import keras.models


class VGG:
    def __init__(self,
                 input_shape=None,
                 filters=64,
                 batch_norm=None,
                 dropout=None,
                 extractor=False,
                 pooling=None,
                 classes=2):
        """
        Initialize VGG architecture neural network
        Args:
            input_shape: shape tuple of the data
                either 2D: (Y, X ,C) or 3D: (Z, Y ,X ,C)
                should not be smaller than 16
            filters: number of filters calculated in the first layer of the network
                the more, the more complex network can become
            batch_norm: momentum of the batch normalization or None if no normalization should happen
                If not None batch norm will be applied after each conv block (before max pool)
                and after each dense layer.
            dropout: dropout rate after each dense layer or None if no dropout should happen
            extractor: whether network should just extract features
                without top classification
            pooling: optional pooling mode for feature extraction
                - None: means that the output of the model will be
                    the 4D tensor
                - avg: means that global average pooling
                    will be applied to the raw output and
                    results will be 2D tensor
                - max: means that global max pooling will
                    be done
            classes: number of classes to predict if extractor is False
        """
        input_dim = len(input_shape) - 1
        assert 2 <= input_dim <= 3

        self.input_shape = input_shape
        # Reconfigure if data is 4D (assumes channel dimension is always there).
        self.kernel_size = (3,) * input_dim
        self.max_pool_size = (2,) * input_dim
        if input_dim == 2:
            self.Conv = keras.layers.Conv2D
            self.MaxPooling = keras.layers.MaxPooling2D
        elif input_dim == 3:
            self.Conv = keras.layers.Conv3D
            self.MaxPooling = keras.layers.MaxPooling3D

        self.first_filters = filters
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.only_extractor = extractor
        self.pooling = pooling
        self.classes = classes

    def add_requested_batch_norm(self, x):
        if self.batch_norm is not None:
            x = keras.layers.BatchNormalization(momentum=self.batch_norm)(x)
        return x

    def add_requested_dropout(self, x):
        if self.dropout is not None:
            x = keras.layers.Dropout(self.dropout)(x)
        return x

    def add_block(self, x, conv_layers, filters, block_name):
        for i in range(1, conv_layers + 1):
            x = self.Conv(filters, self.kernel_size,
                          activation='relu',
                          padding='same',
                          name=block_name + '_conv' + str(i))(x)
        x = self.add_requested_batch_norm(x)
        x = self.MaxPooling(self.max_pool_size, strides=self.max_pool_size, name=block_name + '_pool')(x)
        return x

    def build(self):
        img_input = keras.layers.Input(shape=self.input_shape)

        x = self.add_block(img_input, conv_layers=2, filters=self.first_filters, block_name='block1')
        x = self.add_block(x, conv_layers=2, filters=self.first_filters * 2, block_name='block2')
        x = self.add_block(x, conv_layers=3, filters=self.first_filters * 4, block_name='block3')
        last_filters = self.first_filters * 4
        x = self.add_block(x, conv_layers=3, filters=last_filters, block_name='block4b')

        if not self.only_extractor:
            # Classification block
            x = keras.layers.Flatten(name='flatten')(x)
            # in original it is 8 times bigger than last filter count (512)
            x = keras.layers.Dense(last_filters * 4, activation='relu', name='fc1')(x)
            x = self.add_requested_batch_norm(x)
            x = self.add_requested_dropout(x)
            x = keras.layers.Dense(last_filters * 4, activation='relu', name='fc2')(x)
            x = self.add_requested_batch_norm(x)
            x = self.add_requested_dropout(x)
            if self.classes == 2:
                activation = 'sigmoid'
                predictions = 1
            else:
                activation = 'softmax'
                predictions = self.classes

            x = keras.layers.Dense(predictions, activation=activation, name='predictions')(x)
        else:
            if self.pooling == 'avg':
                x = keras.layers.GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = keras.layers.GlobalMaxPooling2D()(x)

        model = keras.models.Model(img_input, x, name='vgg16')

        return model
