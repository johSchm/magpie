#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
todo:
------------------------------------------- """


import os


def join(path, *paths, win_raw=False):
    """ Joins multiple paths.
    (OS independent)
    :param path:
    :param paths:
    :param win_raw: true:   r'\foo\'
                    false:  '/foo/'
    :return: merged path
    """
    if type(path) is list:
        joined_paths = [join(p, *paths, win_raw=win_raw) for p in path]
    elif type(paths[-1]) is list:
        joined_paths = [join(path, paths[:-1][0], p, win_raw=win_raw) for p in paths[-1]]
    else:
        if win_raw:
            joined_paths = repr(os.path.join(path, *paths))
        else:
            joined_paths = os.path.join(path, *paths).replace("\\", "/")
    return joined_paths


def convert_to_win(path):
    """ Converts a path to the windows path style '\' instead of '/'.
    :param path:
    :return: new path
    """
    if type(path) is not str:
        return path
    path = path.replace('/', '\\')
    return path


def list_files(path):
    """ Returns a list of files in the given directory.
    :param path:
    :return: list of paths
    """
    files = []
    for f in os.listdir(path):
        file_path = join(path, f)
        if os.path.isfile(file_path):
            files.append(f)
    return files


def create_directory(path):
    """ Creates a directory if it doesnt exist yet.
    :param path:
    """
    path = os.path.dirname(path)
    if os.path.exists(path):
        return
    os.makedirs(path)
