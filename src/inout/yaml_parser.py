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


def read(path: str) -> str:
    """ Reads a yaml file and returns the content.
    :param path: (str) The path to the yaml file.
    :return: The content string.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("The file does not exist! Passed path {}".format(path))
    content = ""
    with open(path, 'r') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return content

