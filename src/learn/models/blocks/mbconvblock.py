#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:       https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
@todo:
@bug:
@brief:     MB Conv Block.
------------------------------------------- """


import tensorflow as tf
from tensorflow.keras.layers import Lambda, Conv2D, Activation, Multiply, \
    BatchNormalization, DepthwiseConv2D, Add
import learn.models.layers.swish as sw
import learn.models.blocks.seblock as seb
import learn.models.layers.dropconnect as dc


def MBConvBlock(input_filters, output_filters,
                kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                data_format=None,
                kernel_initializer=None):

    if data_format is None:
        data_format = tf.keras.backend.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):

        if expand_ratio != 1:
            x = Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=kernel_initializer,
                padding='same',
                use_bias=False)(inputs)
            x = BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)
            x = sw.Swish()(x)
        else:
            x = inputs

        x = DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=strides,
            depthwise_initializer=kernel_initializer,
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = sw.Swish()(x)

        if has_se:
            x = seb.SEBlock(input_filters, se_ratio, expand_ratio,
                        data_format)(x)

        # output phase

        x = Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=kernel_initializer,
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = dc.DropConnect(drop_connect_rate)(x)

                x = Add()([x, inputs])

        return x

    return block