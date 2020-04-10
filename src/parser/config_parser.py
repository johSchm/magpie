#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:      The config parser can read and write config files.
------------------------------------------- """


import parser.yaml_parser as yaml
import os


DEFAULT_CONFIG_ROOT_PATH = os.path.join(os.getcwd(), "..", "..", "res", "models")


class ConfigParser:
    """ A config parser.
    """

    def __init__(self, path: str, config_root_path=DEFAULT_CONFIG_ROOT_PATH):
        """ Init. method.
        :param path: (str) The path to the root config file.
        :param config_root_path: (str) The config root path.
        """
        self._config_root_path = config_root_path
        self._path = path

    def valid(self) -> bool:
        """ Checks if the config file is valid.
        :return: (bool)
        """
        try:
            yaml.read(self._path)
        except yaml.YAMLError:
            return False
        return True

    def read(self, accumulate=False, key=None) -> dict:
        """ Reads the config file and accumulates the output of all linked files.
        :param key: (str): The key/link to further config files.
        :param accumulate: (bool): Accumulate the additional content (key) with the base content.
        :return: (str) The accumulated content.
        """
        content = yaml.read(self._path)
        if key is not None:
            additional = yaml.read(os.path.join(
                    self._config_root_path, content["key"], content[key]))
            if accumulate:
                content.update(additional)
            else:
                content = additional
        return content

    def get_full_path(self, folder: str) -> str:
        """ Concatenates the full path for the specified config folder.
        :param folder: The specific folder.
        :return: The absolute path.
        """
        content = yaml.read(self._path)
        full_path = os.path.join(self._config_root_path, content["key"], folder)
        return full_path
