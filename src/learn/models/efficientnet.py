#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
------------------------------------------- """


import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import \
    Dense, Dropout, Activation,\
    Flatten, MaxPooling3D, ZeroPadding3D, \
    Conv2D, Conv3D, MaxPooling2D, \
    MaxPool3D, BatchNormalization, Concatenate, concatenate, \
    Input, concatenate, BatchNormalization, AveragePooling3D, \
    Reshape, Lambda
import learn.models.root_model as model
import learn.models.model_utils as utils
import learn.models.layers.convbn as cb
from enum import Enum



class VersionID(Enum):
    """ Versions of the EfficientNet `BX`, where `BX` refers to the version ID.
    """
    RAW = -1
    DEFAULT = 0
    LATEST = 7
    B0 = 0
    B1 = 1
    B2 = 2
    B3 = 3
    B4 = 4
    B5 = 5
    B6 = 6
    B7 = 7


def efficientnet_factory(
        input_shape: list,
        output_shape: list,
        id=VersionID.LATEST,
        weight_links=None,
        optimizer=utils.Optimizer.ADADELTA,
        loss=utils.Loss.SPARSE_CAT_CROSS_ENTROPY,
        metrics=utils.Metrics.SPARSE_CAT_ACCURACY,
        normalization=utils.Normalizations.BATCH_NORM,
        log_path=None,
        ckpt_path=None,
        parallel=False,
        layer_prefix="",
        load_weights_after_logits=True):
    """ Init. method.

    Args:
        load_weights_after_logits (bool): Load the weights after adding the logits.
        layer_prefix (str): Add this prefix to ALL layer names.
        weight_links (dict): This dictionary contains the link to the weights.
        input_shape (list): Input shape of the input data (W x H x D).
        output_shape (list): The output shape for the output data.
        optimizer (utils.Optimizer): The optimizer.
        loss (utils.Loss): The loss.
        metrics (utils.Metrics): The evaluation metric or metrics.
        normalization (utils.Normalizations): The normalization method.
        parallel (bool): Enable GPU parallelism.
        log_path (str): The path to the desired log directory.
        ckpt_path (str): The path to the checkpoint directory.
        id (VersionID): The version ID of the model.

    Returns:
        model (EfficientNet): The configured model.
    """
    setup_dict = {}
    if id == VersionID.B0 or id == VersionID.B0.value:
        setup_dict = {}

    return EfficientNet(
        input_shape=input_shape,
        output_shape=output_shape,
        weight_links=weight_links,
        optimizer=optimizer,
        loss=loss.metrics,
        normalization=normalization,
        log_path=log_path,
        ckpt_path=ckpt_path,
        parallel=parallel,
        layer_prefix=layer_prefix,
        load_weights_after_logits=load_weights_after_logits,
        kwargs=setup_dict)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def SEBlock(input_filters, se_ratio, expand_ratio, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

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
        x = layers.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = layers.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = Swish()(x)
        # Excite
        x = layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = layers.Activation('sigmoid')(x)
        out = layers.Multiply()([x, inputs])
        return out

    return block


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def MBConvBlock(input_filters, output_filters,
                kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                data_format=None):

    if data_format is None:
        data_format = K.image_data_format()

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
            x = layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=EfficientNetConvInitializer(),
                padding='same',
                use_bias=False)(inputs)
            x = layers.BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)
            x = Swish()(x)
        else:
            x = inputs

        x = layers.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=strides,
            depthwise_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        data_format)(x)

        # output phase

        x = layers.Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)

                x = layers.Add()([x, inputs])

        return x

    return block

class EfficientNet(model.Model):
    """ Efficient Net.
    Code:   https://github.com/titu1994/keras-efficientnets
    Paper:  https://arxiv.org/abs/1905.11946
    """

    def __init__(self,
                 input_shape: list,
                 output_shape: list,
                 weight_links=None,
                 optimizer=utils.Optimizer.ADADELTA,
                 loss=utils.Loss.SPARSE_CAT_CROSS_ENTROPY,
                 metrics=utils.Metrics.SPARSE_CAT_ACCURACY,
                 normalization=utils.Normalizations.BATCH_NORM,
                 log_path=None,
                 ckpt_path=None,
                 parallel=False,
                 layer_prefix="",
                 load_weights_after_logits=True,
                 **kwargs):
        """ Init. method.

        Args:
            load_weights_after_logits (bool): Load the weights after adding the logits.
            layer_prefix (str): Add this prefix to ALL layer names.
            weight_links (dict): This dictionary contains the link to the weights.
            input_shape (list): Input shape of the input data (W x H x D).
            output_shape (list): The output shape for the output data.
            optimizer (utils.Optimizer): The optimizer.
            loss (utils.Loss): The loss.
            metrics (utils.Metrics): The evaluation metric or metrics.
            normalization (utils.Normalizations): The normalization method.
            parallel (bool): Enable GPU parallelism.
            log_path (str): The path to the desired log directory.
            ckpt_path (str): The path to the checkpoint directory.

        Keyword Args:

        """
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            parallel=parallel,
            ckpt_path=ckpt_path,
            log_path=log_path)

        self._layer_prefix = layer_prefix
        self._optimizer = optimizer
        self._metrics = metrics
        self._loss = loss
        self._weight_links = weight_links
        self._norm = normalization
        self._model = self._build_ff_model()
        self._construct_model()
        self._configure(optimizer=optimizer, loss=loss, metrics=metrics)
        if self._weight_links is not None and "ckpt-e" not in self._weight_links["name"]:#not load_weights_after_logits:
            self._model = self._setup_pretrained_weights()
            self._setup_logits_layers()
        else:
            print("Loading weights after adding logits ...")
            self._setup_logits_layers()
            self._model = self._setup_pretrained_weights()

    def _construct_model(self) -> None:
        """ Adds layers to the learn.

        Args:
            self (self): self

        Returns:
            None: None
        """

        if data_format is None:
            data_format = K.image_data_format()

        channel_axis = 1 if self.shape_format == 'channels_first' else -1

        if block_args_list is None:
            block_args_list = get_default_block_list()
        # count number of strides to compute min size
        stride_count = 1
        for block_args in block_args_list:
            if block_args.strides is not None and block_args.strides[0] > 1:
                stride_count += 1

        min_size = int(2 ** stride_count)

        # Determine proper input shape and default size.
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=default_size,
                                          min_size=min_size,
                                          data_format=data_format,
                                          require_flatten=include_top,
                                          weights=weights)

        # Stem part
        if input_tensor is None:
            inputs = layers.Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                inputs = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                inputs = input_tensor

        x = inputs
        x = layers.Conv2D(
            filters=round_filters(32, width_coefficient,
                                  depth_divisor, min_depth),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
        drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

        # Blocks part
        for block_idx, block_args in enumerate(block_args_list):
            assert block_args.num_repeat > 0

            # Update block input and output filters based on depth multiplier.
            block_args.input_filters = round_filters(block_args.input_filters, width_coefficient, depth_divisor,
                                                     min_depth)
            block_args.output_filters = round_filters(block_args.output_filters, width_coefficient, depth_divisor,
                                                      min_depth)
            block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

            # The first block needs to take care of stride and filter size increase.
            x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                            block_args.kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            batch_norm_momentum, batch_norm_epsilon, data_format)(x)

            if block_args.num_repeat > 1:
                block_args.input_filters = block_args.output_filters
                block_args.strides = [1, 1]

            for _ in range(block_args.num_repeat - 1):
                x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                                block_args.kernel_size, block_args.strides,
                                block_args.expand_ratio, block_args.se_ratio,
                                block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                                batch_norm_momentum, batch_norm_epsilon, data_format)(x)

        # Head part
        x = layers.Conv2D(
            filters=round_filters(1280, width_coefficient, depth_coefficient, min_depth),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format)(x)

            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)

            x = layers.Dense(classes, kernel_initializer=EfficientNetDenseInitializer())(x)
            x = layers.Activation('softmax')(x)

        else:
            if pooling == 'avg':
                x = layers.GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = layers.GlobalMaxPooling2D()(x)

        outputs = x

        # Ensure that the models takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)

        model = Model(inputs, outputs)

        elif weights is not None:
            model.load_weights(weights)

        return model

    def _build_ff_model(self):
        """ Builds a feed forward learn.
        :return: ff model
        """
        model_in, model_out = self._construct_model()
        self._model = Model(inputs=model_in, outputs=model_out)
        return self._model  #Sequential()

    def _setup_pretrained_weights(self):
        """ Setup for pretrained weights.
        :return: model
        """
        if self._weight_links is None:
            print("No pre-trained weights loaded ...")
        else:
            print("Loading pre-trained weights ...")
            self._load_weights(self._weight_links)
        return self._model

    def _setup_logits_layers(self):
        """ Setup for logits layers.
        :return: model
        """
        self._model.outputs = [self._model.layers[-1].output]
        output = self._model.get_layer(self._layer_prefix + 'Logits_global_avg_pool').output
        output = Dropout(0.5)(output)
        output = cb.conv3d_bn(output, self._num_classes, 1, 1, 1, padding='same',
                              use_bias=True, use_activation_fn=False, use_bn=False,
                              name=self._layer_prefix + 'Logits_Conv3d_6a_1x1')
        num_frames_remaining = int(output.shape[1])
        output = Reshape((num_frames_remaining, self._num_classes))(output)
        output = Lambda(lambda output: K.mean(output, axis=1, keepdims=False),
                        output_shape=lambda s: (s[0], s[2]))(output)
        output = Activation('softmax', name=self._layer_prefix + 'Logits_prediction')(output)
        new_model = Model(self._model.input, output)
        self._model = new_model
        self._configure(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)
        return new_model
