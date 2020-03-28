#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:
todo:
------------------------------------------- """


import numpy as np
import re


def sort_alphanumerical(l):
    """ Sorts a alpha-numerical list.
    :param l:
    :return: sorted list
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)
    return l


def join(array_1, array_2):
    """ Constructs a column stack.
    :param array_1:
    :param array_2:
    :return: stack
    """
    if len(array_1) != len(array_2):
        raise ValueError("Passed array length is not valid!")
    joined_list = []
    for i, label in enumerate(array_2):
        joined_list.append([label, array_1[i]])
    return np.array(joined_list)


def disjoin(array):
    """ Splits the 2D list into two seperate lists.
    :param array
    :return: array, array
    """
    if len(array) <= 0:
        raise ValueError("Array length is not valid!")
    list_1 = []
    list_2 = []
    for sublist in array:
        list_1.append(sublist[0])
        list_2.append(sublist[1])
    return np.array(list_1), np.array(list_2)


# Python program to match wild card characters

# The main function that checks if two given strings match.
# The first string may contain wildcard characters
# https://www.geeksforgeeks.org/wildcard-character-matching/
def match(first, second):
    # If we reach at the end of both strings, we are done
    if len(first) == 0 and len(second) == 0:
        return True

    # Make sure that the characters after '*' are present
    # in second string. This function assumes that the first
    # string will not contain two consecutive '*'
    if len(first) > 1 and first[0] == '*' and len(second) == 0:
        return False

    # If the first string contains '?', or current characters
    # of both strings match
    if (len(first) > 1 and first[0] == '?') or (len(first) != 0
                                                and len(second) != 0 and first[0] == second[0]):
        return match(first[1:], second[1:]);

        # If there is *, then there are two possibilities
    # a) We consider current character of second string
    # b) We ignore current character of second string.
    if len(first) != 0 and first[0] == '*':
        return match(first[1:], second) or match(first, second[1:])

    return False
