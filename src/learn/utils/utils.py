#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:
todo:
------------------------------------------- """


import tensorflow as tf
import os
from enum import Enum
from keras.utils.data_utils import get_file
import learn.gpu.hvd_wrapper as hvd
import keras.backend as K
import utils.array_utils as array_utils


class ModelID(Enum):
    """ Enum of supported models IDs.
    """
    BIC = "bic"
    BVC = "bvc"
    MCVC = "mcvc"
    C3D = "c3d"
    TS2D = "2d2s"
    TD_VGG = "td_vgg"
    I3D = "i3d"
    T3D = "t3d"
    I3D2S = "i3d2s"
    STLSTM = "stlstm"


class Optimizer(Enum):
    """ Supported Optimizers.
    """
    ADAM = 'adam'
    ADADELTA = 'Adadelta'


class Loss(Enum):
    """ Supported Loss Functions.
    """
    BINARY_CROSS_ENTROPY = 'binary_crossentropy'
    CAT_CROSS_ENTROPY = 'categorical_crossentropy'
    SPARSE_CAT_CROSS_ENTROPY = 'sparse_categorical_crossentropy'
    L1_L2 = 'l1_l2'


class Metrics(Enum):
    """ Supported Metrics.
    """
    ACCURACY = tf.keras.metrics.Accuracy()
    CAT_ACCURACY = tf.keras.metrics.CategoricalAccuracy()
    CAT_CROSS_ENTROPY = tf.metrics.CategoricalCrossentropy()
    PRECISION = tf.keras.metrics.Precision()
    RECALL = tf.keras.metrics.Recall()
    SPARSE_CAT_ACCURACY = tf.metrics.SparseCategoricalAccuracy()
    SPARSE_CAT_CROSS_ENTROPY = tf.metrics.SparseCategoricalCrossentropy()
    AUC = tf.keras.metrics.AUC()
    MSE = tf.keras.metrics.mse


class ValidationSource(Enum):
    """ Enum of validation set sources.
    """

    # use the processed test set as the validation set
    TEST_AS_VAL = "test"

    # split the processed training set into train and validation set
    FROM_TRAIN = "train"


class WeightLinks(Enum):
    """ Enum of possible weight links.
    """
    I3D_RGB_KINECTS = {
        "name": "i3d_rgb_kinects",
        "http": 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
        "path": "../res/weights/i3d/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5"
    }
    I3D_FLOW_KINECTS = {
        "name": "i3d_flow_kinects",
        "http": 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
        "path": "../res/weights/i3d/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5"
    }
    I3D_RGB_IMAGENET_AND_KINECTS = {
        "name": "i3d_rgb_imagenet_and_kinects",
        "http": 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
        "path": "../res/weights/i3d/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5"
    }
    I3D_FLOW_IMAGENET_AND_KINECTS = {
        "name": "i3d_flow_imagenet_and_kinects",
        "http": 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
        "path": "../res/weights/i3d/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5"
    }
    C3D_RGB_SPORTS1M = {
        "name": "c3d_rgb_sports1m",
        "http": "https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5",
        "path": "../res/weights/c3d/c3d-sports1M_weights.h5"
    }
    EFFNETB0_RGB_IMAGENET_TOP = {
        "name": "effnetb0_rgb_imagenet_top",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b0.h5",
        "path": "../res/weights/effnet/effnetb0_top_imagenet_weights.h5",
        "sample_size": 224
    }
    EFFNETB0_RGB_IMAGENET_NOTOP = {
        "name": "effnetb0_rgb_imagenet_notop",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b0_notop.h5",
        "path": "../res/weights/effnet/effnetb0_notop_imagenet_weights.h5",
        "sample_size": 224
    }
    EFFNETB1_RGB_IMAGENET_TOP = {
        "name": "effnetb1_rgb_imagenet_top",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b1.h5",
        "path": "../res/weights/effnet/effnetb1_top_imagenet_weights.h5",
        "sample_size": 240
    }
    EFFNETB1_RGB_IMAGENET_NOTOP = {
        "name": "effnetb1_rgb_imagenet_notop",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b1_notop.h5",
        "path": "../res/weights/effnet/effnetb1_notop_imagenet_weights.h5",
        "sample_size": 240
    }
    EFFNETB2_RGB_IMAGENET_TOP = {
        "name": "effnetb2_rgb_imagenet_top",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b2.h5",
        "path": "../res/weights/effnet/effnetb2_top_imagenet_weights.h5",
        "sample_size": 260
    }
    EFFNETB2_RGB_IMAGENET_NOTOP = {
        "name": "effnetb2_rgb_imagenet_notop",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b2_notop.h5",
        "path": "../res/weights/effnet/effnetb2_notop_imagenet_weights.h5",
        "sample_size": 260
    }
    EFFNETB3_RGB_IMAGENET_TOP = {
        "name": "effnetb3_rgb_imagenet_top",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b3.h5",
        "path": "../res/weights/effnet/effnetb3_top_imagenet_weights.h5",
        "sample_size": 300
    }
    EFFNETB3_RGB_IMAGENET_NOTOP = {
        "name": "effnetb3_rgb_imagenet_notop",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b3_notop.h5",
        "path": "../res/weights/effnet/effnetb3_notop_imagenet_weights.h5",
        "sample_size": 300
    }
    EFFNETB4_RGB_IMAGENET_TOP = {
        "name": "effnetb4_rgb_imagenet_top",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b4.h5",
        "path": "../res/weights/effnet/effnetb4_top_imagenet_weights.h5",
        "sample_size": 380
    }
    EFFNETB4_RGB_IMAGENET_NOTOP = {
        "name": "effnetb4_rgb_imagenet_notop",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b4_notop.h5",
        "path": "../res/weights/effnet/effnetb4_notop_imagenet_weights.h5",
        "sample_size": 380
    }
    EFFNETB5_RGB_IMAGENET_TOP = {
        "name": "effnetb5_rgb_imagenet_top",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b5.h5",
        "path": "../res/weights/effnet/effnetb5_top_imagenet_weights.h5",
        "sample_size": 456
    }
    EFFNETB5_RGB_IMAGENET_NOTOP = {
        "name": "effnetb5_rgb_imagenet_notop",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b5_notop.h5",
        "path": "../res/weights/effnet/effnetb5_notop_imagenet_weights.h5",
        "sample_size": 456
    }
    EFFNETB6_RGB_IMAGENET_TOP = {
        "name": "effnetb6_rgb_imagenet_top",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b6.h5",
        "path": "../res/weights/effnet/effnetb6_top_imagenet_weights.h5",
        "sample_size": 528
    }
    EFFNETB6_RGB_IMAGENET_NOTOP = {
        "name": "effnetb6_rgb_imagenet_notop",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b6_notop.h5",
        "path": "../res/weights/effnet/effnetb6_notop_imagenet_weights.h5",
        "sample_size": 528
    }
    EFFNETB7_RGB_IMAGENET_TOP = {
        "name": "effnetb7_rgb_imagenet_top",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b7.h5",
        "path": "../res/weights/effnet/effnetb7_top_imagenet_weights.h5",
        "sample_size": 600
    }
    EFFNETB7_RGB_IMAGENET_NOTOP = {
        "name": "effnetb7_rgb_imagenet_notop",
        "http": "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b7_notop.h5",
        "path": "../res/weights/effnet/effnetb7_notop_imagenet_weights.h5",
        "sample_size": 600
    }


class Normalizations:
    """ Supported Normalizations.
    """
    BATCH_NORM = 'batch_norm'


def get_weight_dict(weight_name, checkpoint_path=None):
    """ returns the weight dict (see WeightLinks)
    :param weight_name:
    :param checkpoint_path: only req if loading own checkpoint
    :return: weight dict
    """
    if weight_name is None:
        return None
    if type(weight_name) is list:
        return [get_weight_dict(wname) for wname in weight_name]
    if array_utils.match("*ckpt-e*.hdf5", weight_name):
        full_ckpt_path = os.path.join(checkpoint_path, weight_name)
        if not os.path.exists(full_ckpt_path):
            raise FileNotFoundError("Checkpoints not found!")
        return {
            "name": weight_name,
            "http": None,
            "path": full_ckpt_path
        }
    if weight_name == WeightLinks.C3D_RGB_SPORTS1M.value["name"]:
        return WeightLinks.C3D_RGB_SPORTS1M.value
    elif weight_name == WeightLinks.I3D_RGB_KINECTS.value["name"]:
        return WeightLinks.I3D_RGB_KINECTS.value
    elif weight_name == WeightLinks.I3D_FLOW_KINECTS.value["name"]:
        return WeightLinks.I3D_FLOW_KINECTS.value
    elif weight_name == WeightLinks.I3D_RGB_IMAGENET_AND_KINECTS.value["name"]:
        return WeightLinks.I3D_RGB_IMAGENET_AND_KINECTS.value
    elif weight_name == WeightLinks.I3D_FLOW_IMAGENET_AND_KINECTS.value["name"]:
        return WeightLinks.I3D_FLOW_IMAGENET_AND_KINECTS.value
    else:
        raise ValueError("Weight name not found. See @WeightLinks")


def init_optimizer(optimizer_dict):
    """ Initiates an optimizer object based on an optimizer dictionary.
    :param optimizer_dict:
    :return: optimizer
    """
    if type(optimizer_dict) is not dict:
        return
    optimizer = None
    if optimizer_dict["name"] == "Adam":
        lr = optimizer_dict["learning_rate"]
        beta_1 = optimizer_dict["beta_1"]
        beta_2 = optimizer_dict["beta_2"]
        amsgrad = optimizer_dict["amsgrad"]
        optimizer = tf.keras.optimizers.Adam(
            hvd.wrap_learning_rate(lr),
            beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)
    elif optimizer_dict["name"] == "Adadelta":
        lr = optimizer_dict["learning_rate"]
        rho = optimizer_dict["rho"]
        epsilon = optimizer_dict["epsilon"]
        optimizer = tf.keras.optimizers.Adadelta(
            hvd.wrap_learning_rate(lr),
            rho=rho, epsilon=epsilon)
    elif optimizer_dict["name"] == "SGD":
        lr = optimizer_dict["learning_rate"]
        momentum = optimizer_dict["momentum"]
        nesterov = optimizer_dict["nesterov"]
        optimizer = tf.keras.optimizers.SGD(
            hvd.wrap_learning_rate(lr),
            momentum=momentum, nesterov=nesterov)
    return optimizer


def get_loss_func(key_loss):
    """ Returns the loss function for a specific key.
    :param key_loss:
    :return: loss func
    """
    if key_loss == Loss.L1_L2:
        return l1_l2_loss
    return key_loss


def get_metric_key(metrics_as_str):
    """ Returns the metric key for a string.
    :param metrics_as_str:
    :return: key
    """
    metrics = []
    for metric_as_str in metrics_as_str:
        if metric_as_str == "SPARSE_CAT_ACCURACY":
            metric = Metrics.SPARSE_CAT_ACCURACY.value
        elif metric_as_str == "ACCURACY":
            metric = Metrics.ACCURACY.value
        elif metric_as_str == "CAT_ACCURACY":
            metric = Metrics.CAT_ACCURACY.value
        elif metric_as_str == "CAT_CROSS_ENTROPY":
            metric = Metrics.CAT_CROSS_ENTROPY.value
        elif metric_as_str == "PRECISION":
            metric = Metrics.PRECISION.value
        elif metric_as_str == "RECALL":
            metric = Metrics.RECALL.value
        elif metric_as_str == "SPARSE_CAT_CROSS_ENTROPY":
            metric = Metrics.SPARSE_CAT_CROSS_ENTROPY.value
        elif metric_as_str == "AUC":
            metric = Metrics.AUC.value
        elif metric_as_str == "MSE":
            metric = Metrics.MSE.value
        else:
            metric = None
        metrics.append(metric)
    return metrics


def download_weights(weights_links):
    """ This will download the weights to the path in the dict.
    :param weights_links:
    """
    path = os.path.join(os.getcwd(), weights_links['path'])
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    get_file(path, weights_links['http'], cache_subdir='models',
             md5_hash='b7a93b2f9156ccbebe3ca24b41fc5402')


def setup_db(raw_db, shuffle=1000, batch=10):
    """ Setup a raw DB.
    :param raw_db:
    :param shuffle:
    :param batch:
    :return: adapted DB
    """
    if raw_db is None:
        return None
    raw_db = raw_db.repeat()
    raw_db = raw_db.shuffle(shuffle)
    raw_db = raw_db.batch(batch)
    raw_db = raw_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return raw_db


def l1_l2_loss(target, pred):
    """ Computes the L1 L2 loss.
    :param target:
    :param pred:
    :return: loss
    """
    diff = target - pred
    loss_ = tf.pow(diff, 2) + tf.abs(diff) # L2 + L1
    return K.mean(loss_, axis=list(range(5)))


def get_strategy(parallel=False, quiet=True, log=False):
    """ Sets the scope for parallel GPU usage, if enabled.
    https://www.tensorflow.org/guide/gpu
    https://www.tensorflow.org/guide/distributed_training
    :param parallel
    :param quiet:
    :param log:
    :return scope
    """
    if log:
        tf.debugging.set_log_device_placement(True)
    if parallel:
        strategy = tf.distribute.MirroredStrategy()
    else:
        #strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
        strategy = None
    if not quiet:
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    return strategy


def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier.

    Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    multiplier = float(width_coefficient)
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)
