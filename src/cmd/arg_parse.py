#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:      A helper and config file for argument parsing.
------------------------------------------- """


import argparse


def parse_args():
    """ Main arg parser.
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', default="", type=str, metavar='N',
                        help='The path to the data.')
    parser.add_argument('-t', '--training', default=False, type=bool, metavar='N',
                        help='Initiate a new training phase.')
    parser.add_argument('-c', '--classify', default=False, type=bool, metavar='N',
                        help='Classifies the data under the provided data path.')
    args = parser.parse_args()
    return args
