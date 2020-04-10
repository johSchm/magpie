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
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K


class ArcFaceLayer(Layer):
    """ The ArcFace layer.
    Implements the additive angular margin loss, proposed in the paper.
    """

    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        """ Init. method.
        :param n_classes: (int) The number of classes.
        :param s: (float) Feature rescale value.
        :param m: (float) Target logits cosine offset.
        :param regularizer: (Regularizer) The regularizer.
        :param kwargs:
        """
        super(ArcFaceLayer, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def get_config(self):
        """ Override the get_config method.
        :return: config
        """
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        })
        return config

    def build(self, input_shape):
        """ Builds the layer with its weights.
        :param input_shape: (list) The input shape.
        """
        super(ArcFaceLayer, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        """ Calls the layer object.
        :param inputs: (x,y) Input values.
        :return: The pooled logits.
        """
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        """ Returns the output shape
        :param input_shape: (list) The input shape.
        :return: output shape
        """
        return None, self.n_classes