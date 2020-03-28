#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:       https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
@todo:
@bug:
@brief:
------------------------------------------- """


import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
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


def get_width_coefficient(vid: VersionID) -> float:
    """ Returns the width coefficient for the specific model version:

    Args:
        vid (VersionID): Version ID.

    Return:
        width coefficient: float
    """
    if type(vid) == VersionID:
        vid = vid.value
    if vid == VersionID.B0:
        return 1.0
    elif vid == VersionID.B1:
        return 1.0
    elif vid == VersionID.B2:
        return 1.1
    elif vid == VersionID.B3:
        return 1.2
    elif vid == VersionID.B4:
        return 1.4
    elif vid == VersionID.B5:
        return 1.6
    elif vid == VersionID.B6:
        return 1.8
    elif vid == VersionID.B7:
        return 2.0
    else:
        raise ValueError("`vid` has to be from type `VersionID` or a number, which fits `VersionID`.")


def get_depth_coefficient(vid: VersionID) -> float:
    """ Returns the depth coefficient for the specific model version:

    Args:
        vid (VersionID): Version ID.

    Return:
        depth coefficient: float
    """
    if type(vid) == VersionID:
        vid = vid.value
    if vid == VersionID.B0:
        return 1.0
    elif vid == VersionID.B1:
        return 1.1
    elif vid == VersionID.B2:
        return 1.2
    elif vid == VersionID.B3:
        return 1.4
    elif vid == VersionID.B4:
        return 1.8
    elif vid == VersionID.B5:
        return 2.2
    elif vid == VersionID.B6:
        return 2.6
    elif vid == VersionID.B7:
        return 3.1
    else:
        raise ValueError("`vid` has to be from type `VersionID` or a number, which fits `VersionID`.")


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
        shape_format (list): The shape format.

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


class EfficientNetConvInitializer(tf.keras.initializers.Initializer):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas base_path we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    # Arguments:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    # Returns:
      an initialization for the variable
    """
    def __init__(self):
        super(EfficientNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.keras.backend.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.keras.backend.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


class EfficientNetDenseInitializer(tf.keras.initializers.Initializer):
    """Initialization for dense kernels.
        This initialization is equal to
          tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                          distribution='uniform').
        It is written out explicitly base_path for clarity.

        # Arguments:
          shape: shape of variable
          dtype: dtype of variable
          partition_info: unused

        # Returns:
          an initialization for the variable
    """
    def __init__(self):
        super(EfficientNetDenseInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.keras.backend.floatx()

        init_range = 1.0 / np.sqrt(shape[1])
        return tf.keras.backend.random_uniform(shape, -init_range, init_range, dtype=dtype)



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
                 log_path=None,
                 ckpt_path=None,
                 layer_prefix="",
                 data_format=None,
                 **kwargs):
        """ Init. method.

        Args:
            data_format (list): The format of the image shape.
            layer_prefix (str): Add this prefix to ALL layer names.
            weight_links (dict): This dictionary contains the link to the weights.
            input_shape (list): Input shape of the input data (W x H x D).
            output_shape (list): The output shape for the output data.
            optimizer (utils.Optimizer): The optimizer.
            loss (utils.Loss): The loss.
            metrics (utils.Metrics): The evaluation metric or metrics.
            log_path (str): The path to the desired log directory.
            ckpt_path (str): The path to the checkpoint directory.

        Keyword Args:
            block_args (list): Block arguments.
        """
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            ckpt_path=ckpt_path,
            log_path=log_path)

        self._batch_norm_momentum = DEFAULT_BATCH_NORM_MOMENTUM
        self._batch_norm_epsilon = DEFAULT_BATCH_NORM_EPSILON
        self._pooling = "max"
        if kwargs is not None:
            if "block_args" in kwargs.keys():
                self._block_args = None if kwargs["block_args"] is None else bargs.get_default_block_list()
            if "width_coefficient" in kwargs.keys():
                self._width_coefficient = kwargs["width_coefficient"]
            if "depth_coefficient" in kwargs.keys():
                self._depth_coefficient = kwargs["depth_coefficient"]
            if "depth_divisor" in kwargs.keys():
                self._depth_divisor = kwargs["depth_divisor"]
            if "min_depth" in kwargs.keys():
                self._min_depth = kwargs["min_depth"]
            if "batch_norm_momentum" in kwargs.keys():
                self._batch_norm_momentum = kwargs["batch_norm_momentum"]
            if "batch_norm_epsilon" in kwargs.keys():
                self._batch_norm_epsilon= kwargs["batch_norm_epsilon"]
            if "drop_connect_rate" in kwargs.keys():
                self._drop_connect_rate= kwargs["drop_connect_rate"]
            if "pooling" in kwargs.keys():
                self._pooling= kwargs["pooling"]
        self._data_format = None if data_format is None else tf.keras.backend.image_data_format()
        self._channel_axis = 1 if self._data_format == 'channels_first' else -1
        self._layer_prefix = layer_prefix
        self._optimizer = optimizer
        self._metrics = metrics
        self._loss = loss
        self._weight_links = weight_links
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

    def _round_filters(self, filters):
        """ Rounds the number of filters if required.

        Args:
            filters (int): Number of original filters.

        Returns:
            rounded filters (int): Altered number of filters.
        """
        if self._width_coefficient and self._depth_divisor and self._min_depth:
            filters = utils.round_filters(
                filters=filters,
                width_coefficient=self._width_coefficient,
                depth_divisor=self._depth_divisor,
                min_depth=self._min_depth)
        return filters

    def _construct_model(self) -> (tf.Tensor, tf.Tensor):
        """ Adds layers to the learn.

        Args:
            self (self): self

        Returns:
            None: None
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
            filters=self._round_filters(32),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)(x)
        x = sw.Swish()(x)

        num_blocks = sum([block_args.num_repeat for block_args in self._block_args])
        drop_connect_rate_per_block = self._drop_connect_rate / float(num_blocks)

        # Blocks part
        for block_idx, block_args in enumerate(self._block_args):
            assert block_args.num_repeat > 0

            # Update block input and output filters based on depth multiplier.
            block_args.input_filters = self._round_filters(block_args.input_filters)
            block_args.output_filters = self._round_filters(block_args.output_filters)
            block_args.num_repeat = utils.round_repeats(
                block_args.num_repeat, self._depth_coefficient)

            # The first block needs to take care of stride and filter size increase.
            x = mbb.MBConvBlock(block_args.input_filters, block_args.output_filters,
                            block_args.kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            self._batch_norm_momentum, self._batch_norm_epsilon, self._data_format)(x)

            if block_args.num_repeat > 1:
                block_args.input_filters = block_args.output_filters
                block_args.strides = [1, 1]

            for _ in range(block_args.num_repeat - 1):
                x = mbb.MBConvBlock(block_args.input_filters, block_args.output_filters,
                                block_args.kernel_size, block_args.strides,
                                block_args.expand_ratio, block_args.se_ratio,
                                block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                                self._batch_norm_momentum, self._batch_norm_epsilon, self._data_format)(x)

        # Head part
        x = Conv2D(
            filters=self._round_filters(1280),
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

            x = Dense(len(self._output_shape), kernel_initializer=EfficientNetDenseInitializer())(x)
            x = Activation('softmax')(x)

        else:
            if self._pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif self._pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        outputs = x

        # Ensure that the models takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)

        return inputs, outputs

    def _build_model(self) -> Model:
        """ Builds the model

        Returns:
            model: The final model.
        """
        model_in, model_out = self._construct_model()
        self._model = Model(inputs=model_in, outputs=model_out)
        return self._model

    def _setup_pretrained_weights(self) -> Model:
        """ Setup for pretrained weights.
        :return: model
        """
        if self._weight_links is None:
            print("No pre-trained weights loaded ...")
        else:
            print("Loading pre-trained weights ...")
            self._load_weights(self._weight_links)
        return self._model

