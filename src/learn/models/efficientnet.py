#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:       https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
             https://github.com/titu1994/keras-efficientnets
             https://arxiv.org/abs/1905.11946
@todo:
@bug:
@brief:      The efficient net model implementation.
------------------------------------------- """


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Dense, Dropout, Activation,\
    Conv2D, Input, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.utils import get_source_inputs
import learn.root_model as model
import learn.utils.utils as utils
from enum import Enum
import numpy as np
import learn.models.blocks.blockargs as bargs
import learn.models.layers.swish as sw
import learn.models.blocks.mbconvblock as mbb


DEFAULT_DROPOUT_RATE = 0.0
DEFAULT_WIDTH_COEFFICIENT = 0.0
DEFAULT_DEPTH_COEFFICIENT = 0.0
DEFAULT_DEPTH_DIVISOR = 8
DEFAULT_MIN_DEPTH = 0
DEFAULT_BATCH_NORM_MOMENTUM = 0.99
DEFAULT_BATCH_NORM_EPSILON = 1e-3
DEFAULT_DROP_CONNECT_RATE = 0.0


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
        optimizer=utils.Optimizer.ADADELTA.value,
        loss=utils.Loss.SPARSE_CAT_CROSS_ENTROPY.value,
        metrics=utils.Metrics.SPARSE_CAT_ACCURACY.value,
        log_path=None,
        ckpt_path=None,
        layer_prefix=""):
    """ Init. method.
    :param layer_prefix: (str) Add this prefix to ALL layer names.
    :param weight_links: (dict) This dictionary contains the link to the weights.
    :param input_shape: (list) Input shape of the input data (W x H x D).
    :param output_shape: (list) The output shape for the output data.
    :param optimizer: (utils.Optimizer) The optimizer.
    :param loss: (utils.Loss) The loss.
    :param metrics: (utils.Metrics) The evaluation metric or metrics.
    :param log_path: (str) The path to the desired log directory.
    :param ckpt_path: (str) The path to the checkpoint directory.
    :param id: (VersionID) The version ID of the model.
    :return model (EfficientNet): The configured model.
    :raises ValueError if the passed id is not part of VersionID.
            ValueError if the version specific image size does not fit the passed input shape.
    """
    if type(id) == VersionID:
        id = id.value
    if id == VersionID.B0.value:
        width_coefficient = 1.0
        depth_coefficient = 1.0
        default_size = 224
        dropout_rate = 0.2
    elif id == VersionID.B1.value:
        width_coefficient = 1.0
        depth_coefficient = 1.1
        default_size = 240
        dropout_rate = 0.2
    elif id == VersionID.B2.value:
        width_coefficient = 1.1
        depth_coefficient = 1.2
        default_size = 260
        dropout_rate = 0.3
    elif id == VersionID.B3.value:
        width_coefficient = 1.2
        depth_coefficient = 1.4
        default_size = 300
        dropout_rate = 0.3
    elif id == VersionID.B4.value:
        width_coefficient = 1.4
        depth_coefficient = 1.8
        default_size = 380
        dropout_rate = 0.4
    elif id == VersionID.B5.value:
        width_coefficient = 1.6
        depth_coefficient = 2.2
        default_size = 456
        dropout_rate = 0.4
    elif id == VersionID.B6.value:
        width_coefficient = 1.8
        depth_coefficient = 2.6
        default_size = 528
        dropout_rate = 0.5
    elif id == VersionID.B7.value:
        width_coefficient = 2.0
        depth_coefficient = 3.1
        default_size = 600
        dropout_rate = 0.5
    else:
        raise ValueError("{0} is not defined within VersionID!".format(id))
    if default_size not in input_shape:
        raise ValueError("The image size defined in the `input_shape`"
                         " has to be {0} and not the size contained in {1}!"
                         .format(default_size, input_shape))
    return EfficientNet(
        dropout_rate=dropout_rate,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        input_shape=input_shape,
        output_shape=output_shape,
        weight_links=weight_links,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        log_path=log_path,
        ckpt_path=ckpt_path,
        layer_prefix=layer_prefix,
        block_args=bargs.get_default_blockargs())


class EfficientNetConvInitializer(tf.keras.initializers.Initializer):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas base_path we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.
    """
    def __init__(self):
        """ Init. method.
        """
        super(EfficientNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        """ Call method.
        :param shape:
        :param dtype:
        :return: an initialization for the variable
        """
        dtype = dtype or tf.keras.backend.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.keras.backend.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


class EfficientNetDenseInitializer(tf.keras.initializers.Initializer):
    """Initialization for dense kernels.
        This initialization is equal to
          tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out', distribution='uniform').
        It is written out explicitly base_path for clarity.
    """
    def __init__(self):
        """ Init. method.
        """
        super(EfficientNetDenseInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        """ Call method.
        :param shape:
        :param dtype:
        :return: an initialization for the variable
        """
        dtype = dtype or tf.keras.backend.floatx()

        init_range = 1.0 / np.sqrt(shape[1])
        return tf.keras.backend.random_uniform(shape, -init_range, init_range, dtype=dtype)


class EfficientNet(model.Model):
    """ Efficient Net.
    """

    def __init__(self,
                 input_shape: list,
                 output_shape: list,
                 weight_links=None,
                 optimizer=utils.Optimizer.ADADELTA.value,
                 loss=utils.Loss.SPARSE_CAT_CROSS_ENTROPY.value,
                 metrics=utils.Metrics.SPARSE_CAT_ACCURACY.value,
                 log_path=None,
                 ckpt_path=None,
                 layer_prefix="",
                 data_format=None,
                 block_args=None,
                 width_coefficient=DEFAULT_WIDTH_COEFFICIENT,
                 depth_coefficient=DEFAULT_DEPTH_COEFFICIENT,
                 batch_norm_momentum=DEFAULT_BATCH_NORM_MOMENTUM,
                 batch_norm_epsilon=DEFAULT_BATCH_NORM_EPSILON,
                 drop_connect_rate=DEFAULT_DROP_CONNECT_RATE,
                 dropout_rate=DEFAULT_DROPOUT_RATE,
                 **kwargs):
        """ Init. method.
        :param data_format (list): The format of the image shape.
        :param layer_prefix (str): Add this prefix to ALL layer names.
        :param weight_links (dict): This dictionary contains the link to the weights.
        :param input_shape (list): Input shape of the input data (W x H x D).
        :param output_shape (list): The output shape for the output data.
        :param optimizer (utils.Optimizer): The optimizer.
        :param loss (utils.Loss): The loss.
        :param metrics (utils.Metrics): The evaluation metric or metrics.
        :param log_path (str): The path to the desired log directory.
        :param ckpt_path (str): The path to the checkpoint directory.
        :param block_args (list): Block arguments.
        :param width_coefficient (float):
        :param depth_coefficient (float):
        :param batch_norm_momentum (float):
        :param batch_norm_epsilon (float):
        :param drop_connect_rate (float):
        :param dropout_rate (float):
        """
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            ckpt_path=ckpt_path,
            log_path=log_path)

        self._batch_norm_momentum = DEFAULT_BATCH_NORM_MOMENTUM
        self._batch_norm_epsilon = DEFAULT_BATCH_NORM_EPSILON
        self._drop_connect_rate = DEFAULT_DROP_CONNECT_RATE
        self._block_args = []
        self._include_top = True
        self._dropout_rate = dropout_rate
        self._block_args = block_args if block_args is not None else bargs.get_default_blockargs()
        self._width_coefficient = width_coefficient
        self._depth_coefficient = depth_coefficient
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._drop_connect_rate = drop_connect_rate
        self._data_format = data_format if data_format is not None else tf.keras.backend.image_data_format()
        self._channel_axis = 1 if self._data_format == 'channels_first' else -1
        self._layer_prefix = layer_prefix
        self._optimizer = optimizer
        self._metrics = metrics
        self._loss = loss
        self._weight_links = weight_links
        self._depth_divisor = DEFAULT_DEPTH_DIVISOR
        self._min_depth = DEFAULT_MIN_DEPTH
        self._model = self._build_model()
        self._configure(optimizer=optimizer, loss=loss, metrics=metrics)
        #if self._weight_links is not None and "ckpt-e" not in self._weight_links["name"]:#not load_weights_after_logits:
        #    self._model = self._setup_pretrained_weights()
        #    self._setup_logits_layers()
        #else:
        #    print("Loading weights after adding logits ...")
        #    self._setup_logits_layers()
        #    self._model = self._setup_pretrained_weights()

    def _construct_model(self) -> (tf.Tensor, tf.Tensor):
        """ Adds layers to the learn.
        :return input and output tensor
        """
        input_tensor = None
        if input_tensor is None:
            inputs = Input(shape=self._input_shape)
        else:
            if not tf.is_tensor(input_tensor):
                inputs = Input(tensor=input_tensor, shape=self._input_shape)
            else:
                inputs = input_tensor

        x = inputs
        x = Conv2D(
                filters=utils.round_filters(32, self._width_coefficient, self._depth_divisor, self._min_depth),
                kernel_size=[3, 3], strides=[2, 2], kernel_initializer=EfficientNetConvInitializer(),
                padding='same', use_bias=False)(x)
        x = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)(x)
        x = sw.Swish()(x)

        if self._block_args is not None and len(self._block_args) > 0:
            num_blocks = sum([block_args.num_repeat for block_args in self._block_args])
            drop_connect_rate_per_block = self._drop_connect_rate / float(num_blocks)

            # Blocks part
            for block_idx, block_args in enumerate(self._block_args):
                assert block_args.num_repeat > 0

                # Update block input and output filters based on depth multiplier.
                block_args.input_filters = utils.round_filters(
                    block_args.input_filters, self._width_coefficient, self._depth_divisor, self._min_depth)
                block_args.output_filters = utils.round_filters(
                    block_args.output_filters, self._width_coefficient, self._depth_divisor, self._min_depth)
                block_args.num_repeat = utils.round_repeats(
                    block_args.num_repeat, self._depth_coefficient)

                # The first block needs to take care of stride and filter size increase.
                x = mbb.MBConvBlock(
                    block_args.input_filters, block_args.output_filters,
                    block_args.kernel_size, block_args.strides,
                    block_args.expand_ratio, block_args.se_ratio,
                    block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                    self._batch_norm_momentum, self._batch_norm_epsilon, self._data_format)(x)

                if block_args.num_repeat > 1:
                    block_args.input_filters = block_args.output_filters
                    block_args.strides = [1, 1]

                for _ in range(block_args.num_repeat - 1):
                    x = mbb.MBConvBlock(
                        block_args.input_filters, block_args.output_filters,
                        block_args.kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        self._batch_norm_momentum, self._batch_norm_epsilon, self._data_format)(x)

        # Head part
        x = Conv2D(
            filters=utils.round_filters(1280, self._width_coefficient, self._depth_divisor, self._min_depth),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)(x)
        x = sw.Swish()(x)

        if self._include_top:
            x = GlobalAveragePooling2D(data_format=self._data_format)(x)
            if self._dropout_rate > 0:
                x = Dropout(self._dropout_rate)(x)
            x = Dense(self._output_shape[0], kernel_initializer=EfficientNetDenseInitializer())(x)
            x = Activation('softmax')(x)
        else:
            #x = GlobalAveragePooling2D()(x)
            x = GlobalMaxPooling2D()(x)
        outputs = x

        # Ensure that the models takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)

        return inputs, outputs

    def _build_model(self) -> Model:
        """ Builds the model
        :return model: The composed model.
        """
        model_in, model_out = self._construct_model()
        self._model = Model(inputs=model_in, outputs=model_out)
        return self._model

    def _setup_weights(self) -> Model:
        """ Setup for weights.
        :return: model
        """
        if self._weight_links is None:
            print("No pre-trained weights loaded ...")
        else:
            print("Loading pre-trained weights ...")
            self._load_weights(self._weight_links)
        return self._model

