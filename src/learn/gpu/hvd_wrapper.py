#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@brief:     A Horovod wrapper for horovod usage.
            This enables parallel GPU usage.
@refs:
@todo:
------------------------------------------- """


import math
import tensorflow as tf
from sys import platform

# linux
if platform == "linux" or platform == "linux2":
    import horovod.tensorflow.keras as hvd
# OS X
elif platform == "darwin":
    import horovod.tensorflow.keras as hvd
# Windows...
elif platform == "win32":
    hvd = None


def init(en_mem_growth=False, set_visible_dev=False):
    """ This initializes the horovod package.
    :param en_mem_growth:
    :param set_visible_dev:
    """
    if hvd is not None:
        hvd.init()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            if en_mem_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
        if gpus and set_visible_dev:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    else:
        print("Horovod not supported on this system!")


def is_initialized():
    """ Checks if horovod is initialized.
    :return: bool
    """
    try:
        hvd.size()
    except ValueError:
        return False
    return True


def get_gpu_num():
    """ Returns the number of supported GPUs.
    :return: num of gpus
    """
    if hvd is None or not is_initialized():
        return 1
    return hvd.size()


def wrap_learning_rate(backbone_learning_rate):
    """ Wraps a learning rate for parallel GPU usage.
    :param backbone_learning_rate:
    :return: wrapped learning rate
    """
    if hvd is None or not is_initialized():
        return backbone_learning_rate
    learning_rate = backbone_learning_rate * hvd.size()
    return learning_rate


def wrap_optimizer(backbone_optimizer):
    """ Wraps a optimizer for parallel GPU usage.
    :param backbone_optimizer:
    :return: wrapped optimizer
    """
    if hvd is None or not is_initialized():
        return backbone_optimizer
    optimizer = hvd.DistributedOptimizer(backbone_optimizer)
    return optimizer


def wrap_dataset(backbone_dataset):
    """ Wraps a dataset for parallel GPU usage.
    :param backbone_dataset:
    :return: wrapped datatset
    """
    if hvd is None or not is_initialized():
        return backbone_dataset
    dataset = backbone_dataset.shard(hvd.size(), hvd.rank())
    return dataset


def wrap_epochs(backbone_epochs):
    """ Wraps a epochs for parallel GPU usage.
    :param backbone_epochs:
    :return: wrapped epochs
    """
    if hvd is None or not is_initialized():
        return backbone_epochs
    epochs = int(math.ceil(backbone_epochs / hvd.size()))
    return epochs


def restrict_to_gpu(gpu_id):
    """ restricts the resource allocation to a specified gpu.
    :param gpu_id:
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            print("Set visibility for GPU " + str(gpus[gpu_id]))
        except RuntimeError as e:
            print(e)
    print("Successfully applied GPU {} restriction.".format(gpu_id))
