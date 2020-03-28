#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:
todos:
------------------------------------------- """


import glob
import cv2
from enum import Enum
import random
import shutil
import os
from string import ascii_lowercase


LETTERS = [letter for letter in ascii_lowercase]

DATA_INDEXING = {
    "a": "RGB",
    "b": "optical_flow",
    "c": "optical_flow_lk",
    "d": "optical_flow_tvl1",
    "e": "pose",
    "f": "segmentation"
}


class MemoryType(Enum):
    """ Supported memory types.
    """
    VIRTUAL = "virtual"
    PHYSICAL = "physical"
    VIRTUAL_AND_PHYSICAL = "virtual_and_physical"


class ReturnTypes(Enum):
    """ Supported return types for loading stored data.
    """
    DATASET = 0
    GENERATOR = 1
    RAW_DATA = 2


class DataPattern(Enum):
    """ Supported return patterns for the data.
    XY_XY:      Train(data + label) and Test(data + label)
    X_X_Y_Y:    Train(data), Test(data), Train(label), and Test(label)
    TF_SAMPLE:  Predefined TF Record feature structure.
    """
    XY_XY = 0
    X_X_Y_Y = 1
    TF_RECORD = 2


def delete_data(path):
    """ Deletes data in an given path.
    :param path:
    """
    if os.path.isdir(path):
        if not is_dir_empty(path):
            try:
                shutil.rmtree(path)
            except PermissionError as e:
                pass
    elif os.path.isfile(path):
        os.remove(path)
    else:
        pass
        # raise ValueError("{} is not a valid file or directory!".format(path))


def is_dir_empty(path):
    """ Boolean function for checking if a directory is empty or not.
    :param path:
    :return: True, False
    """
    if len(os.listdir(path)) == 0:
        return True
    return False


def largest_set(sets):
    """ Returns the largest sets index.
    :param sets
    :return index
    """
    if sets is None or type(sets) is not list or len(sets) <= 0:
        return None
    if len(sets) == 1:
        return 0
    m, max_len = 0, 0
    for i, _set in enumerate(sets):
        if len(_set) > max_len:
            m, max_len = i, len(_set)
    return m


def smallest_set(dictionary):
    """ Returns the smallest sets index.
    :param dictionary
    :return index
    """
    if type(dictionary) is not dict or len(dictionary) <= 0:
        return None
    m, min_len = 0, 0
    for key, _set in dictionary.items():
        if len(_set) < min_len or min_len == 0:
            m, min_len = key, len(_set)
    return m


def downsize_set(_set, target_len):
    """ Downsize a set to a specific target length.
    :param _set:
    :param target_len:
    :return: reduced set
    """
    if _set is None or len(_set) <= 0 or type(target_len) is not int:
        return None
    if target_len <= 0:
        return []
    while len(_set) != target_len:
        idx = random.randint(0, len(_set) - 1)
        del _set[idx]
    return _set


def shared_items(dict_1, dict_2):
    """ Returns the shared items of two dictionaries.
    :param dict_1:
    :param dict_2:
    :return: shared items
    """
    if type(dict_1) is not dict or type(dict_2) is not dict:
        return None
    return {k: dict_1[k] for k in dict_1 if k in dict_2 and dict_1[k] == dict_2[k]}


def equal_values(dictionary):
    """ Checks if all values in a dictionary are equal.
    :param dictionary:
    :return: boolean
    """
    if type(dictionary) is not dict:
        return False
    i = 0
    master_value = 0
    for key, value in dictionary.items():
        if i == 0:
            master_value = value
        if value != master_value:
            return False
        i += 1
    return True


def value_redundancy(dictionary):
    """ Returns if a value is already in the dictionary.
    :param dictionary:
    :return: True or False
    """
    if type(dictionary) is not dict:
        return False
    for key_1, value_1 in dictionary.items():
        for key_2, value_2 in dictionary.items():
            if value_1 == value_2 and key_1 != key_2:
                return True
    return False


def extract_vector_from_matrix(matrix, idx):
    """ Extracts a vector from a matrix (2D List).
    :param matrix:
    :param idx of vector
    :return: vector
    """
    if type(matrix) is not list or type(idx) is not int:
        return None
    vector = []
    for v in matrix:
        vector.append(v[idx])
    return vector


def get_ds_length(dataset=None, path=None, path_pattern=None):
    """ Returns the length of a given dataset.
    :param dataset: priority 1
    :param path:    priority 2
    :param path_pattern:
    :return: length
    """
    if dataset is not None:
        length = 0
        for _ in dataset:
            length += 1
        return length
    elif path is not None:
        count = 0
        if path_pattern is not None:
            paths = glob.glob(os.path.join(path, path_pattern))
        else:
            paths = os.listdir(path)
        for name in paths:
            if os.path.isfile(os.path.join(path, name)):
                count += 1
        return count
    else:
        raise ValueError("Nothing specified for the length measurement.")


def char_to_alpha_pos(char):
    """ Maps a character to tht position in the alphabet.
    :param char: char
    :return: pos
    """
    char = char.lower()
    pos = [i for i, x in enumerate(LETTERS) if x == char]
    return pos[0]


def alpha_pos_to_char(pos):
    """ Maps a alphabet position to character.
    :param pos: pos
    :return: char
    """
    if pos < 0 or pos >= len(LETTERS):
        raise ValueError("Position in alphabet not defined!")
    char = LETTERS[pos]
    return char


def merge_datasets(root_ds, other_ds):
    """ Joins two datasets.
    :param root_ds
    :param other_ds
    :return merged_ds
    """
    if root_ds is None:
        return None
    if other_ds is None:
        return root_ds
    # not yet implemented
