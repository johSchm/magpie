#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       2020
refs:
todos:
------------------------------------------- """


import numpy as np
import cv2


def txt_in_img(img, txt, position=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX,
               size=1, color=(209, 80, 0, 255), stroke=3):
    """ Puts text on top of image.
    :param img:
    :param txt:
    :param position: eg: (10, 50)
    :param font:
    :param size:
    :param color:
    :param stroke:
    :return image
    """
    if img is None:
        return
    if txt is None or txt == '':
        return img
    if type(txt) is not str:
        txt = str(txt)
    if type(img) is str:
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    cv2.putText(img, txt, position, font, size, color, stroke)
    return img


def get_best_class(result):
    """ Finds the best guess in the result list (highest accuracy).
    :param result:
    :return: class
    """
    max_val = ["", 0.0]
    for class_name in result.keys():
        if result[class_name] > max_val[1]:
            max_val = [class_name, result[class_name]]
    return max_val[0]
