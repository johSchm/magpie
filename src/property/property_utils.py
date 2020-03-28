#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
------------------------------------------- """


import os
import json
import utils.path_utils as path_utils
import importlib.util


MASTER_PATH = os.path.join(os.getcwd(), "../res/properties/master.json")


def load_hyper_param(hyper_param_path):
    """ Loads / imports the hyper parameter.
    :param hyper_param_path:
    """
    import learn.utils.utils as model_utils
    import tensorflow as tf
    import learn.gpu.hvd_wrapper as hvd

    LOSS = model_utils.Loss.SPARSE_CAT_CROSS_ENTROPY
    OPTIMIZER = tf.keras.optimizers.Adam(
        hvd.wrap_learning_rate(0.05),
        beta_1=0.85, beta_2=0.9, amsgrad=False)
    METRICS = [
        model_utils.Metrics.SPARSE_CAT_ACCURACY,
        # model_utils.Metrics.ACCURACY,
        # model_utils.Metrics.AUC,
        # model_utils.Metrics.CAT_ACCURACY,
        # model_utils.Metrics.CAT_CROSS_ENTROPY,
        # model_utils.Metrics.SPARSE_CAT_CROSS_ENTROPY,
        # model_utils.Metrics.PRECISION,
        # model_utils.Metrics.RECALL
    ]


def get_settings_file_path(
        master_path=None, auto_search=True,
        auto_iteration=0, max_iterations=5):
    """ Returns the path to the settings file.
    :param master_path:
    :param auto_search:
    :param auto_iteration:
    :param max_iterations:
    :return: path
    """
    data = None
    if master_path is None:
        master_path = MASTER_PATH
    f = None
    try:
        f = open(master_path, "r")
        data = json.load(f)
    except FileNotFoundError:
        if auto_iteration >= max_iterations:
            auto_search = False
        if auto_search:
            new_path = path_utils.join('..', master_path)
            auto_iteration += 1
            get_settings_file_path(
                master_path=new_path,
                auto_iteration=auto_iteration)
        else:
            raise FileNotFoundError("File not found!")
    if f is not None:
        f.close()
    return data


def load_json(path):
    """ Maps the category list to a dictionary with int flags.
    :param path
    :return: dictionary
    """
    if os.stat(path).st_size == 0:
        return None
    file = open(path)
    content = json.load(file)
    file.close()
    return content


def get_num_of_items(path):
    """ Returns the number of items in an json file.
    :param path:
    :return: number (int)
    """
    content = load_json(path)
    return len(content)


def read_json_item(path, item=None):
    """ Returns the value of a properties item.
    :param path: path to the json file
    :param item: If None, this will return the entire file content.
    :return: value
    """
    if os.stat(path).st_size == 0:
        return None
    file = open(path)
    content = json.load(file)
    file.close()
    if item is None:
        return content
    try:
        return content.get(item)
    except KeyError as e:
        print("Key {} not found!".format(item))
        return None
