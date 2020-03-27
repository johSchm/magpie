#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:       https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
@todo:
@bug:
@brief:     Contains tools to create universal blocks for ML models.
------------------------------------------- """


import tensorflow as tf
from tensorflow.keras.layers import Lambda, Conv2D, Activation, Multiply
import learn.models.layers.swish


def SEBlock(input_filters, se_ratio, expand_ratio, data_format=None, kernel_initializer=None):
    """ SEBlock.

    Args:
        input_filters:
        se_ratio:
        expand_ratio:
        data_format:
        kernel_initializer:
    """

    if data_format is None:
        data_format = tf.keras.backend.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = Lambda(lambda a: tf.keras.backend.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=kernel_initializer,
            padding='same',
            use_bias=True)(x)
        x = Swish()(x)
        # Excite
        x = Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=kernel_initializer,
            padding='same',
            use_bias=True)(x)
        x = Activation('sigmoid')(x)
        out = Multiply()([x, inputs])
        return out

    return block
