#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:     Test for Dlib Face Detector.
------------------------------------------- """


import unittest
import learn.models.dlibfacedetect as fd
from PIL import Image
import numpy as np
import learn.utils.utils as utils


class TestDlibFaceDetectorHOG(unittest.TestCase):
    """ Test for Dlib Face Detector HOG-based.
    """

    def test_prediction(self):
        """ Construction the model and test it with an example image.
        """
        model = fd.DlibHOGFaceDetector()
        with self.subTest():
            self.assertEqual(type(model), fd.DlibHOGFaceDetector)
        with self.subTest():
            img = Image.open("../../res/dataset/test_image_barack_obama.jpg")
            pred = model.apply(np.array(img))
            self.assertEqual(len(pred), 1)


class TestDlibFaceDetectorCNN(unittest.TestCase):
    """ Test for Dlib Face Detector CNN-based.
    """

    def test_prediction(self):
        """ Construction the model and test it with an example image.
        """
        model = fd.DlibCNNFaceDetector(utils.WeightLinks.DLIB_FACE_DETECTION_DEFAULT)
        with self.subTest():
            self.assertEqual(type(model), fd.DlibCNNFaceDetector)
        with self.subTest():
            img = Image.open("../../res/dataset/test_image_barack_obama.jpg")
            pred = model.apply(np.array(img))
            self.assertEqual(len(pred), 1)


if __name__ == '__main__':
    unittest.main()

