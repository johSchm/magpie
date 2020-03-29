#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:     Test for EfficientNet.
------------------------------------------- """


import unittest
from keras_applications.imagenet_utils import decode_predictions
from keras_preprocessing.image import img_to_array, load_img
from keras_applications.imagenet_utils import preprocess_input as _preprocess
import learn.models.efficientnet as en
import learn.utils.utils as utils
import os
import numpy as np
import tensorflow as tf


def get_predictions(model) -> tf.Tensor:
    """ Returns the mapped prediction output for a test sample.
    :param model:
    :return: (Tensor) prediction.
    """
    size = model.get_input_shape()[1]
    filename = os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'dataset', '565727409_61693c5e14.jpg')
    image = img_to_array(load_img(filename, target_size=(size, size)))
    batch = _preprocess(image, None, mode='torch', backend=tf.keras.backend)
    pred = decode_predictions(model.predict(batch), backend=tf.keras.backend, utils=utils)
    return pred


class TestEfficientNetPureNoTop(unittest.TestCase):
    """ Test for EfficientNet without weights and without top included.
    """

    def test_construction_b0_notop(self):
        """ Construction test of the B0 model + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[224, 224, 3],
            output_shape=[1000],
            id=en.VersionID.B0,
            include_top=False)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b1_notop(self):
        """ Construction test of the B1 model+ no top.
        """
        model = en.efficientnet_factory(
            input_shape=[240, 240, 3],
            output_shape=[1000],
            id=en.VersionID.B1,
            include_top=False)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b2_notop(self):
        """ Construction test of the B2 model + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[260, 260, 3],
            output_shape=[1000],
            id=en.VersionID.B2,
            include_top=False)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b3_notop(self):
        """ Construction test of the B3 model + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[300, 300, 3],
            output_shape=[1000],
            id=en.VersionID.B3,
            include_top=False)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b4_notop(self):
        """ Construction test of the B4 model + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[380, 380, 3],
            output_shape=[1000],
            id=en.VersionID.B4,
            include_top=False)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b5_notop(self):
        """ Construction test of the B5 model + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[456, 456, 3],
            output_shape=[1000],
            id=en.VersionID.B5,
            include_top=False)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b6_notop(self):
        """ Construction test of the B6 model + notop.
        """
        model = en.efficientnet_factory(
            input_shape=[528, 528, 3],
            output_shape=[1000],
            id=en.VersionID.B6,
            include_top=False)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b7_notop(self):
        """ Construction test of the B7 model + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[600, 600, 3],
            output_shape=[1000],
            id=en.VersionID.B7,
            include_top=False)
        self.assertEqual(type(model), en.EfficientNet)


class TestEfficientNetPureTop(unittest.TestCase):
    """ Test for EfficientNet without weights and with top included.
    """

    def test_construction_b0(self):
        """ Construction test of the B0 model.
        """
        model = en.efficientnet_factory(
            input_shape=[224, 224, 3],
            output_shape=[1000],
            id=en.VersionID.B0)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b1(self):
        """ Construction test of the B1 model.
        """
        model = en.efficientnet_factory(
            input_shape=[240, 240, 3],
            output_shape=[1000],
            id=en.VersionID.B1)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b2(self):
        """ Construction test of the B2 model.
        """
        model = en.efficientnet_factory(
            input_shape=[260, 260, 3],
            output_shape=[1000],
            id=en.VersionID.B2)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b3(self):
        """ Construction test of the B3 model.
        """
        model = en.efficientnet_factory(
            input_shape=[300, 300, 3],
            output_shape=[1000],
            id=en.VersionID.B3)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b4(self):
        """ Construction test of the B4 model.
        """
        model = en.efficientnet_factory(
            input_shape=[380, 380, 3],
            output_shape=[1000],
            id=en.VersionID.B4)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b5(self):
        """ Construction test of the B5 model.
        """
        model = en.efficientnet_factory(
            input_shape=[456, 456, 3],
            output_shape=[1000],
            id=en.VersionID.B5)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b6(self):
        """ Construction test of the B6 model.
        """
        model = en.efficientnet_factory(
            input_shape=[528, 528, 3],
            output_shape=[1000],
            id=en.VersionID.B6)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b7(self):
        """ Construction test of the B7 model.
        """
        model = en.efficientnet_factory(
            input_shape=[600, 600, 3],
            output_shape=[1000],
            id=en.VersionID.B7)
        self.assertEqual(type(model), en.EfficientNet)


class TestEfficientNetWeightsNoTop(unittest.TestCase):
    """ Test for EfficientNet with weights and without top included.
    """

    def test_construction_b0_weights_notop(self):
        """ Construction test of the B0 model + weights + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[224, 224, 3],
            output_shape=[1000],
            id=en.VersionID.B0,
            include_top=False,
            weight_links=utils.WeightLinks.EFFNETB0_RGB_IMAGENET_NOTOP)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b1_weights_notop(self):
        """ Construction test of the B1 model + weights + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[240, 240, 3],
            output_shape=[1000],
            id=en.VersionID.B1,
            include_top=False,
            weight_links=utils.WeightLinks.EFFNETB1_RGB_IMAGENET_NOTOP)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b2_weights_notop(self):
        """ Construction test of the B2 model + weights + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[260, 260, 3],
            output_shape=[1000],
            id=en.VersionID.B2,
            include_top=False,
            weight_links=utils.WeightLinks.EFFNETB2_RGB_IMAGENET_NOTOP)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b3_weights_notop(self):
        """ Construction test of the B3 model + weights + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[300, 300, 3],
            output_shape=[1000],
            id=en.VersionID.B3,
            include_top=False,
            weight_links=utils.WeightLinks.EFFNETB3_RGB_IMAGENET_NOTOP)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b4_weights_notop(self):
        """ Construction test of the B4 model + weights + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[380, 380, 3],
            output_shape=[1000],
            id=en.VersionID.B4,
            include_top=False,
            weight_links=utils.WeightLinks.EFFNETB4_RGB_IMAGENET_NOTOP)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b5_weights_notop(self):
        """ Construction test of the B5 model + weights + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[456, 456, 3],
            output_shape=[1000],
            id=en.VersionID.B5,
            include_top=False,
            weight_links=utils.WeightLinks.EFFNETB5_RGB_IMAGENET_NOTOP)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b6_weights_notop(self):
        """ Construction test of the B6 model + weights + notop.
        """
        model = en.efficientnet_factory(
            input_shape=[528, 528, 3],
            output_shape=[1000],
            id=en.VersionID.B6,
            include_top=False,
            weight_links=utils.WeightLinks.EFFNETB6_RGB_IMAGENET_NOTOP)
        self.assertEqual(type(model), en.EfficientNet)

    def test_construction_b7_weights_notop(self):
        """ Construction test of the B7 model + weights + no top.
        """
        model = en.efficientnet_factory(
            input_shape=[600, 600, 3],
            output_shape=[1000],
            id=en.VersionID.B7,
            include_top=False,
            weight_links=utils.WeightLinks.EFFNETB7_RGB_IMAGENET_NOTOP)
        self.assertEqual(type(model), en.EfficientNet)


class TestEfficientNetWeightsTop(unittest.TestCase):
    """ Test for EfficientNet with weights and with top included.
    """

    def test_construction_b0_weights(self):
        """ Construction test of the B0 model + weights.
        """
        model = en.efficientnet_factory(
            input_shape=[224, 224, 3],
            output_shape=[1000],
            id=en.VersionID.B0,
            weight_links=utils.WeightLinks.EFFNETB0_RGB_IMAGENET_TOP)
        with self.subTest():
            self.assertEqual(type(model), en.EfficientNet)
        with self.subTest():
            self.assertEqual(get_predictions(model)[0][0][1], 'tiger_cat')

    def test_construction_b1_weights(self):
        """ Construction test of the B1 model + weights.
        """
        model = en.efficientnet_factory(
            input_shape=[240, 240, 3],
            output_shape=[1000],
            id=en.VersionID.B1,
            weight_links=utils.WeightLinks.EFFNETB1_RGB_IMAGENET_TOP)
        with self.subTest():
            self.assertEqual(type(model), en.EfficientNet)
        with self.subTest():
            self.assertEqual(get_predictions(model)[0][0][1], 'tiger_cat')

    def test_construction_b2_weights(self):
        """ Construction test of the B2 model + weights.
        """
        model = en.efficientnet_factory(
            input_shape=[260, 260, 3],
            output_shape=[1000],
            id=en.VersionID.B2,
            weight_links=utils.WeightLinks.EFFNETB2_RGB_IMAGENET_TOP)
        with self.subTest():
            self.assertEqual(type(model), en.EfficientNet)
        with self.subTest():
            self.assertEqual(get_predictions(model)[0][0][1], 'tiger_cat')

    def test_construction_b3_weights(self):
        """ Construction test of the B3 model + weights.
        """
        model = en.efficientnet_factory(
            input_shape=[300, 300, 3],
            output_shape=[1000],
            id=en.VersionID.B3,
            weight_links=utils.WeightLinks.EFFNETB3_RGB_IMAGENET_TOP)
        with self.subTest():
            self.assertEqual(type(model), en.EfficientNet)
        with self.subTest():
            self.assertEqual(get_predictions(model)[0][0][1], 'tiger_cat')

    def test_construction_b4_weights(self):
        """ Construction test of the B4 model + weights.
        """
        model = en.efficientnet_factory(
            input_shape=[380, 380, 3],
            output_shape=[1000],
            id=en.VersionID.B4,
            weight_links=utils.WeightLinks.EFFNETB4_RGB_IMAGENET_TOP)
        with self.subTest():
            self.assertEqual(type(model), en.EfficientNet)
        with self.subTest():
            self.assertEqual(get_predictions(model)[0][0][1], 'tiger_cat')

    def test_construction_b5_weights(self):
        """ Construction test of the B5 model + weights.
        """
        model = en.efficientnet_factory(
            input_shape=[456, 456, 3],
            output_shape=[1000],
            id=en.VersionID.B5,
            weight_links=utils.WeightLinks.EFFNETB5_RGB_IMAGENET_TOP)
        with self.subTest():
            self.assertEqual(type(model), en.EfficientNet)
        with self.subTest():
            self.assertEqual(get_predictions(model)[0][0][1], 'tiger_cat')

    def test_construction_b6_weights(self):
        """ Construction test of the B6 model + weights.
        """
        model = en.efficientnet_factory(
            input_shape=[528, 528, 3],
            output_shape=[1000],
            id=en.VersionID.B6,
            weight_links=utils.WeightLinks.EFFNETB6_RGB_IMAGENET_TOP)
        with self.subTest():
            self.assertEqual(type(model), en.EfficientNet)
        with self.subTest():
            self.assertEqual(get_predictions(model)[0][0][1], 'tiger_cat')

    def test_construction_b7_weights(self):
        """ Construction test of the B7 model + weights.
        """
        model = en.efficientnet_factory(
            input_shape=[600, 600, 3],
            output_shape=[1000],
            id=en.VersionID.B7,
            weight_links=utils.WeightLinks.EFFNETB7_RGB_IMAGENET_TOP)
        with self.subTest():
            self.assertEqual(type(model), en.EfficientNet)
        with self.subTest():
            self.assertEqual(get_predictions(model)[0][0][1], 'tiger_cat')


if __name__ == '__main__':
    unittest.main()

