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
import numpy as np
from tqdm import tqdm


def crop_all_faces(src_path, dst_path, detector):
    """ Crops all faces for all images within a given path.
    :param src_path: (str) The source path to the original images.
    :param dst_path: (str) The destination path to the cropped images.
    :param detector: The face detector.
    """
    if type(src_path) is not str:
        raise TypeError("The source path needs to be a string! Got {} instead!".format(type(src_path)))
    if type(dst_path) is not str:
        raise TypeError("The destination path needs to be a string! Got {} instead!".format(type(dst_path)))
    if not os.path.exists(src_path):
        raise ValueError("The provided source path does not exist! Provided path {}.".format(src_path))
    if not os.path.exists(dst_path):
        print("Destination path {} does not exist. Creating required directory ...")
        os.mkdir(dst_path)
    if type(detector) is not DlibHOGFaceDetector and type(detector) is not DlibCNNFaceDetector:
        raise TypeError("The detector needs to be either {0} or {1}, but got {2} instead!".
                        format(DlibHOGFaceDetector, DlibCNNFaceDetector, type(detector)))
    for image_id, image_name in enumerate(tqdm(os.listdir(src_path))):
        image_src_path = os.path.join(src_path, image_name)
        src_image = Image.open(image_src_path)
        face_bbox = detector.apply(np.array(src_image))
        if len(face_bbox) > 0:
            dst_images = crop_face(src_image, face_bbox)
            for face_id, dst_image in enumerate(dst_images):
                image_dst_path = os.path.join(dst_path, str(image_id) + "_" + str(face_id) + ".jpeg")
                dst_image.save(image_dst_path)
        src_image.close()


def crop_face(image, bbox) -> list:
    """ Crops the face from the original image.
    :param image: original image
    :param bbox: bounding box of the face
    :return: the modified image list
    """
    #if type(image) is not PIL:
    #    raise TypeError("Pillow image required! Got instead " + str(type(image)))
    if type(bbox) is not dlib.rectangles:
        raise TypeError("Bounding box needs to be a dlib.rectangles! Got instead " + str(type(bbox)))
    if len(bbox) <= 0:
        raise ValueError("Bounding Box is empty!")
    cropped_faces = []
    for bb in bbox:
        area = (bb.left(), bb.top(), bb.right(), bb.bottom())
        cropped_img = image.crop(area)
        cropped_faces.append(cropped_img)
    return cropped_faces


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
