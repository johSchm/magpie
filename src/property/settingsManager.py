#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
------------------------------------------- """


import json
import os
import property.property_utils as utils
import utils.path_utils as path_utils


class SettingsManager:
    """ A properties manager.
    """

    def __init__(self, path=None, model_idx=0, settings_idx=0):
        """ Init method.
        :param path:
        :param settings_idx:
        :param model_idx: for multiple settings paths
        """
        self.path = path
        if path is None:
            path = utils.get_settings_file_path()
            if type(path) is dict:
                path = path.get('settings_' + str(settings_idx))[model_idx]
            self.path = path_utils.join('..', path)

    def write(self, item, value, overwrite=False, force=True):
        """ Writes into the properties file.
        :param item:
        :param value:
        :param overwrite:
        :param force:
        """
        data = None
        if not overwrite:
            data = self.read()
        if force:
            path_utils.create_directory(self.path)
            file = open(self.path, 'w+')
        else:
            file = open(self.path, 'w')
        if data is not None:
            data[item] = value
            self._pretty_dump(data, file)
        else:
            self._pretty_dump({item: value}, file)
        file.close()

    @staticmethod
    def _pretty_dump(data, file):
        """ Pretty print into json file.
        :param data:
        :param file:
        """
        if data is not None:
            json.dump(data, file, sort_keys=True, indent=4, separators=(',', ': '))

    def read(self, item=None):
        """ Returns the value of a properties item.
        :param item: If None, this will return the entire file content.
        :return: value
        """
        if not os.path.exists(self.path) or os.stat(self.path).st_size == 0:
            return None
        file = open(self.path)
        content = json.load(file)
        file.close()
        if item is None:
            return content
        try:
            return content.get(item)
        except KeyError as e:
            print("Key {} not found!".format(item))
            return None

    def clear(self):
        """ Clears the file.
        """
        open(self.path, 'w').close()

    def print(self):
        """ Pretty prints the content of the file.
        """
        file = open(self.path)
        content = json.load(file)
        print(json.dumps(content, indent=4, sort_keys=True))
        file.close()
