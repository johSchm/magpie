#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:      A parser for YAML files.
------------------------------------------- """


import yaml
import os


def read(path: str) -> dict:
    """ Reads a yaml file and returns the content.
    :param path: (str) The path to the yaml file.
    :return: The content dict.
    :raises yaml.YAMLError if yaml file not valid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("The file does not exist! Passed path {}".format(path))
    content = ""
    with open(path, 'r') as stream:
        content = yaml.safe_load(stream)
    return content

