#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:      The face detector by dlib.
------------------------------------------- """


import PIL
import cv2
import dlib
import argparse
import time
import learn.utils.utils as utils
import os
import utils.path_utils as path_utils
from PIL import Image


def crop_face(image, bbox) -> Image:
    """ Crops the face from the original image.
    :param image: original image
    :param bbox: bounding box of the face
    :return: the modified image
    """
    if type(image) is not Image.Image and type(image) is not PIL.JpegImagePlugin.JpegImageFile:
        raise TypeError("Pillow image required! Got instead " + str(type(image)))
    if type(bbox) is not dlib.rectangles:
        raise TypeError("Bounding box needs to be a dlib.rectangles! Got instead " + str(type(bbox)))
    if len(bbox) <= 0:
        raise ValueError("Bounding Box is empty!")
    for bb in bbox:
        area = (bb.left(), bb.top(), bb.right(), bb.bottom())
        cropped_img = image.crop(area)
        cropped_img.show()
        pass


class DlibHOGFaceDetector:
    """ Dlib HOB-based face detector
    """

    def __init__(self):
        """ Init. method.

        initialize hog + svm based face detector
        """
        self._hog_face_detector = dlib.get_frontal_face_detector()

    def apply(self, image):
        """ Applies the face detection algorithm.
        :param image:
        :return: HOG faces
        """
        faces_hog = self._hog_face_detector(image, 1)
        x, y, w, h = 0, 0, 0, 0
        face_param = []
        for face in faces_hog:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return faces_hog


class DlibCNNFaceDetector:
    """ Dlib CNN-based face detector
    """

    def __init__(self, weight_links):
        """ Init. method.

        initialize cnn based face detector with the weights

        :param weight_links: (dict) Weights.
        """
        self._weight_links = weight_links
        weights_path = self._extract_weight_path()
        self._cnn_face_detector = dlib.cnn_face_detection_model_v1(weights_path)

    def _extract_weight_path(self):
        """ Extracts the weight path from the links.
        :return: (str) weights path
        """
        if type(self._weight_links) is utils.WeightLinks:
            self._weight_links = self._weight_links.value
        path = os.path.join(path_utils.get_root_path(), self._weight_links['path'])
        if not os.path.isfile(path):
            utils.download_weights(self._weight_links)
        return self._weight_links["path"]

    def apply(self, image):
        """ Applies the face detection algorithm.
        :param image:
        :return: CNN faces
        """
        faces_cnn = self._cnn_face_detector(image, 1)
        for face in faces_cnn:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return faces_cnn
