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
        """
        super().__init__(input_shape=input_shape, output_shape=output_shape,
                         parallel=parallel, ckpt_path=ckpt_path, log_path=log_path)

        if type(input_shape) is dict:
            self.__input_shape = input_shape["RGB"]
        else:
            self.__input_shape = input_shape
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
        include_top = False  # refers to num of classes, disable if diff num of classes then pre-train
        endpoint_logit = True
        end_points = {}

        tf.keras.backend.set_image_data_format('channels_last')
        if tf.keras.backend.image_data_format() == 'channels_first':
            concat_axis = 1
        else:
            concat_axis = 4

        _input = Input(
            shape=self.__input_shape, dtype='float32', name=self._layer_prefix+'input')

        # ===========================================================================================

        # Downsampling via convolution (spatial and temporal)
        end_point = self._layer_prefix + 'Conv3d_1a_7x7'
        net = cb.conv3d_bn(_input, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', name=end_point)
        end_points[end_point] = net

        # Downsampling (spatial only)
        end_point = self._layer_prefix + 'MaxPool3d_2a_3x3'
        net = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same', name=end_point)(net)
        end_points[end_point] = net
        end_point = self._layer_prefix + 'Conv3d_2b_1x1'
        net = cb.conv3d_bn(net, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name=end_point)
        end_points[end_point] = net
        end_point = self._layer_prefix + 'Conv3d_2c_3x3'
        net = cb.conv3d_bn(net, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name=end_point)
        end_points[end_point] = net

        # Downsampling (spatial only)
        end_point = self._layer_prefix + 'MaxPool3d_3a_3x3'
        net = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same', name=end_point)(net)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'Mixed_3b'

        branch = '_Branch_0'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_0 = cb.conv3d_bn(net, 64, 1, 1, 1, padding='same', name=name)

        branch = 'Branch_1'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_1 = cb.conv3d_bn(net, 96, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_1 = cb.conv3d_bn(branch_1, 128, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_2'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_2 = cb.conv3d_bn(net, 16, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_2 = cb.conv3d_bn(branch_2, 32, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_3'
        name = end_point + branch + '_MaxPool3d_0a_3x3'
        branch_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name=name)(net)
        name = end_point + branch + '_Conv3d_0b_1x1'
        branch_3 = cb.conv3d_bn(branch_3, 32, 1, 1, 1, padding='same', name=name)

        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=end_point)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'Mixed_3c'

        branch = 'Branch_0'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_0 = cb.conv3d_bn(net, 128, 1, 1, 1, padding='same', name=name)

        branch = 'Branch_1'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_1 = cb.conv3d_bn(net, 128, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_1 = cb.conv3d_bn(branch_1, 192, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_2'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_2 = cb.conv3d_bn(net, 32, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_2 = cb.conv3d_bn(branch_2, 96, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_3'
        name = end_point + branch + '_MaxPool3d_0a_3x3'
        branch_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name=name)(net)
        name = end_point + branch + '_Conv3d_0b_1x1'
        branch_3 = cb.conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name=name)

        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=end_point)
        end_points[end_point] = net

        # ===========================================================================================

        end_point = self._layer_prefix + 'MaxPool3d_4a_3x3'
        net = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', name=end_point)(net)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'Mixed_4b'

        branch = 'Branch_0'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_0 = cb.conv3d_bn(net, 192, 1, 1, 1, padding='same', name=name)

        branch = 'Branch_1'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_1 = cb.conv3d_bn(net, 96, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_1 = cb.conv3d_bn(branch_1, 208, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_2'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_2 = cb.conv3d_bn(net, 16, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_2 = cb.conv3d_bn(branch_2, 48, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_3'
        name = end_point + branch + '_MaxPool3d_0a_3x3'
        branch_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name=name)(net)
        name = end_point + branch + '_Conv3d_0b_1x1'
        branch_3 = cb.conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name=name)

        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=end_point)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'Mixed_4c'

        branch = 'Branch_0'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_0 = cb.conv3d_bn(net, 160, 1, 1, 1, padding='same', name=name)

        branch = 'Branch_1'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_1 = cb.conv3d_bn(net, 112, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_1 = cb.conv3d_bn(branch_1, 224, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_2'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_2 = cb.conv3d_bn(net, 24, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_2 = cb.conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_3'
        name = end_point + branch + '_MaxPool3d_0a_3x3'
        branch_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name=name)(net)
        name = end_point + branch + '_Conv3d_0b_1x1'
        branch_3 = cb.conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name=name)

        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=end_point)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'Mixed_4d'

        branch = 'Branch_0'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_0 = cb.conv3d_bn(net, 128, 1, 1, 1, padding='same', name=name)

        branch = 'Branch_1'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_1 = cb.conv3d_bn(net, 128, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_1 = cb.conv3d_bn(branch_1, 256, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_2'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_2 = cb.conv3d_bn(net, 24, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_2 = cb.conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_3'
        name = end_point + branch + '_MaxPool3d_0a_3x3'
        branch_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name=name)(net)
        name = end_point + branch + '_Conv3d_0b_1x1'
        branch_3 = cb.conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name=name)

        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=end_point)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'Mixed_4e'

        branch = 'Branch_0'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_0 = cb.conv3d_bn(net, 112, 1, 1, 1, padding='same', name=name)

        branch = 'Branch_1'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_1 = cb.conv3d_bn(net, 144, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_1 = cb.conv3d_bn(branch_1, 288, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_2'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_2 = cb.conv3d_bn(net, 32, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_2 = cb.conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_3'
        name = end_point + branch + '_MaxPool3d_0a_3x3'
        branch_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name=name)(net)
        name = end_point + branch + '_Conv3d_0b_1x1'
        branch_3 = cb.conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name=name)

        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=end_point)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'Mixed_4f'

        branch = 'Branch_0'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_0 = cb.conv3d_bn(net, 256, 1, 1, 1, padding='same', name=name)

        branch = 'Branch_1'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_1 = cb.conv3d_bn(net, 160, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_1 = cb.conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_2'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_2 = cb.conv3d_bn(net, 32, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_2 = cb.conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_3'
        name = end_point + branch + '_MaxPool3d_0a_3x3'
        branch_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name=name)(net)
        name = end_point + branch + '_Conv3d_0b_1x1'
        branch_3 = cb.conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name=name)

        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=end_point)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'MaxPool3d_5a_2x2'
        net = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name=end_point)(net)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'Mixed_5b'

        branch = 'Branch_0'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_0 = cb.conv3d_bn(net, 256, 1, 1, 1, padding='same', name=name)

        branch = 'Branch_1'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_1 = cb.conv3d_bn(net, 160, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_1 = cb.conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_2'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_2 = cb.conv3d_bn(net, 32, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_2 = cb.conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_3'
        name = end_point + branch + '_MaxPool3d_0a_3x3'
        branch_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name=name)(net)
        name = end_point + branch + '_Conv3d_0b_1x1'
        branch_3 = cb.conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name=name)

        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=end_point)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'Mixed_5c'

        branch = 'Branch_0'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_0 = cb.conv3d_bn(net, 384, 1, 1, 1, padding='same', name=name)

        branch = 'Branch_1'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_1 = cb.conv3d_bn(net, 192, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_1 = cb.conv3d_bn(branch_1, 384, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_2'
        name = end_point + branch + '_Conv3d_0a_1x1'
        branch_2 = cb.conv3d_bn(net, 48, 1, 1, 1, padding='same', name=name)
        name = end_point + branch + '_Conv3d_0b_3x3'
        branch_2 = cb.conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name=name)

        branch = 'Branch_3'
        name = end_point + branch + '_MaxPool3d_0a_3x3'
        branch_3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name=name)(net)
        name = end_point + branch + '_Conv3d_0b_1x1'
        branch_3 = cb.conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name=name)

        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=end_point)
        end_points[end_point] = net

        # ===========================================================================================
        end_point = self._layer_prefix + 'Logits'

        if include_top:
            net = AveragePooling3D(pool_size=(2, 7, 7), strides=(1, 1, 1),
                                   padding='valid', name=end_point+'_global_avg_pool')(net)
            net = Dropout(0.5)(net)
            net = cb.conv3d_bn(net, self._num_classes, 1, 1, 1, padding='same',
                               use_bias=True, use_activation_fn=False, use_bn=False,
                               name=end_point+'_Conv3d_6a_1x1')

            num_frames_remaining = int(net.shape[1])
            net = Reshape((num_frames_remaining, self._num_classes))(net)

            # net (raw scores for each class)
            net = Lambda(lambda net: K.mean(net, axis=1, keepdims=False),
                         output_shape=lambda s: (s[0], s[2]))(net)

            if not endpoint_logit:
                net = Activation('softmax', name=end_point+'_prediction')(net)
        else:
            h = int(net.shape[2])
            w = int(net.shape[3])
            net = AveragePooling3D((2, h, w), strides=(1, 1, 1),
                                   padding='valid', name=end_point+'_global_avg_pool')(net)

        return _input, net

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
