#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:
todo:
------------------------------------------- """


import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import BaseLogger
import utils.path_utils as path_utils
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


def setup_callbacks(log_path=None, ckpt_path=None):
    """ Setup method for the training callbacks.
    :return: callbacks
    """
    cbs = []
    if log_path is not None:
        cb = get_checkpoint_cb(ckpt_path)
        cbs.append(cb)
    if ckpt_path is not None:
        cb = get_tensorboard_cb(log_path)
        cbs.append(cb)
    return cbs


def get_tensorboard_cb(path):
    """ Sets the callback and the overall logger.
    :param path
    :return callbacks
    """
    if path is None:
        return
    tb = TensorBoard(log_dir=path, histogram_freq=1)
    BaseLogger(stateful_metrics=None)
    return tb


def get_checkpoint_cb(path):
    """ This will setup a models checkpoint callback.
    Therefore the models is stored after every epoch.
    :param path:
    :return: cb
    """
    if path is None:
        return
    checkpoint_path = path + str()
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    file_path = path_utils.join(
        checkpoint_path,
        "ckpt-e{epoch:02d}.hdf5")
    checkpoint_callback = ModelCheckpoint(
        file_path, verbose=1,
        save_best_only=False, save_weights_only=False,
        save_frequency=1)
    return checkpoint_callback


def log_confusion_matrix(path, epoch, classes, ground_truth, predictions):
    """ Logs the confusion matrix.
    :param path:
    :param epoch:
    :param classes:
    :param ground_truth:
    :param predictions:
    """
    file_writer = tf.summary.create_file_writer(path_utils.join(path + 'cm'))
    con_mat = tf.math.confusion_matrix(labels=ground_truth, predictions=predictions).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    with file_writer.as_default():
        tf.summary.image("Confusion Matrix", image, step=epoch)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path)
    #cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
