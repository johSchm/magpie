#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:       https://github.com/4uiiurz1/keras-arcface
             https://arxiv.org/abs/1801.07698
@todo:
@bug:
@brief:      The arcnet model implementation.
------------------------------------------- """


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten, Layer
from tensorflow.keras import regularizers
import learn.root_model as model
import learn.utils.utils as utils
import learn.models.blocks.vgg as vgg
import learn.models.layers.arcface as afl
import numpy as np


DEFAULT_WEIGHT_DECAY = 1e-4


def generator_batch(db, categories):
    """ A generator for ArcFace training and validation batches.
    :param categories: The list of all categories.
    :param db: The dataset.
    :return (yield) data
    """
    while True:
        for data in db:
            imgs = np.array([item for item in data[0]])
            labels = np.array([item for item in data[1]])
            label_dict = np.array([categories for _ in range(len(data[1]))])
            yield [imgs, label_dict], labels


class ArcFace(model.Model):
    """ The ArcFace network.
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
                 include_top=True,
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
        :param include_top (bool): Include top layers.
        """
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            ckpt_path=ckpt_path,
            log_path=log_path)

        self._include_top = include_top
        self._layer_prefix = layer_prefix
        self._optimizer = optimizer
        self._metrics = metrics
        self._loss = loss
        self._weight_links = weight_links
        self._model = self._build_model()
        self._model = self._setup_weights()
        self._configure(optimizer=optimizer, loss=loss, metrics=metrics)

    def _construct_model(self) -> (tf.Tensor, tf.Tensor):
        """ Adds layers to the learn.
        :return input and output tensor
        """
        _input = Input(shape=self.get_input_shape())
        y = Input(shape=self.get_output_shape())
        x = vgg.vgg_block(_input, 16, 2)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = vgg.vgg_block(x, 32, 2)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = vgg.vgg_block(x, 64, 2)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(512, kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(DEFAULT_WEIGHT_DECAY))(x)
        x = BatchNormalization()(x)
        output = afl.ArcFaceLayer(
            self.get_output_shape()[0], regularizer=regularizers.l2(DEFAULT_WEIGHT_DECAY))([x, y])
        return [_input, y], output

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
