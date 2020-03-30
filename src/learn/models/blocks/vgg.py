#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:       https://github.com/4uiiurz1/keras-arcface
@todo:
@bug:
@brief:      Various VGG blocks.
------------------------------------------- """


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten, Layer
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K


DEFAULT_WEIGHT_DECAY = 1e-4


def vgg_block(x, filters, layers) -> tf.Tensor:
    """ A standard VGG block structure.
    :param x: (layer) The previous layer.
    :param filters: (int) Number of filters.
    :param layers: (int) Number of layers.
    :return: Output tensor (end of VGG block).
    """
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(DEFAULT_WEIGHT_DECAY))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x


def vgg8(args) -> Model:
    """ The VGG8 model.
    :param args: {num_features}: The number of features.
    :return: The constructed model.
    """
    _input = Input(shape=(28, 28, 1))
    x = vgg_block(_input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(DEFAULT_WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    output = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(DEFAULT_WEIGHT_DECAY))(x)

    return Model(_input, output)
