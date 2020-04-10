#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:      These methods manage the face recognition process.
------------------------------------------- """


import learn.models.arcface as af
import parser.config_parser as config
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import learn.utils.utils as utils


class FaceRecognizer:
    """ This class manages the face recognition process.
    """

    def __init__(self, config_file_path):
        """ Init. method.
        :param config_file_path: (str) The path to the config file.
        """
        self._config_parser = config.ConfigParser(config_file_path)
        self._base_config = self._config_parser.read()
        self._hyper_param = self._config_parser.read(key="hyper_parameter_path")
        self._classes = self._config_parser.read(key="class_path")
        self._model = af.ArcFace(
            input_shape=self._hyper_param["input_shape"],
            output_shape=[len(self._classes)],
            log_path=self._config_parser.get_full_path(self._base_config["log_path"]),
            ckpt_path=self._config_parser.get_full_path(self._base_config["checkpoint_path"]),
            optimizer=utils.instantiate_optimizer(self._hyper_param["optimizer"]),
            metrics=utils.get_metric_key(self._hyper_param["metrics"]),
            loss=utils.get_loss_func(self._hyper_param["loss"]),
        )

    @staticmethod
    def _parse_function(data: list, shape: list) -> tuple:
        """ Parse function, which converts the data of a dataset into a appropriate format for training.
        :param data: (list) [sample path, label]
        :param shape: (list) Shape for resizing.
        :return: [sample, label]
        """
        data = data.numpy()
        img = Image.open(data[0].decode("utf-8")).resize(shape)
        img = tf.cast(np.array(img) / 255.0, tf.float32)
        label = tf.cast(int(data[1].decode("utf-8")), tf.int8)
        return img, label

    def train(self, path):
        """ Trains the face recognition model based on the provided face images (path).
        :param path: (str) Path to face images.
        """
        data = []
        for cls in self._classes.keys():
            class_path = os.path.join(path, cls)
            for file in os.listdir(class_path):
                data.append([os.path.join(class_path, file), int(self._classes[cls])])
        data = np.array(data)
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(len(data))
        dataset = dataset.map(lambda x: tf.py_function(
            self._parse_function, [x, self._hyper_param["input_shape"][:-1]], [tf.float32, tf.int8]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self._model.print()
        self._model.train(
            dataset, validation_db=dataset,
            epochs=self._hyper_param["epochs"],
            batch_size=self._hyper_param["batch_size"],
            class_catalog=list(self._classes.values()),
            batch_generator=af.generator_batch)
