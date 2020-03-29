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
import learn.models.efficientnet as en


class TestEfficientNet(unittest.TestCase):
    """ Test for EfficientNet.
    """

    def test_construction_b0(self):
        """ Construction test of the B0 model.
        """
        model = en.efficientnet_factory(
            input_shape=[224, 224, 3],
            output_shape=[1000],
            id=en.VersionID.B0)
        assert type(model) is en.EfficientNet

    def test_construction_b1(self):
        """ Construction test of the B1 model.
        """
        model = en.efficientnet_factory(
            input_shape=[240, 240, 3],
            output_shape=[1000],
            id=en.VersionID.B1)
        assert type(model) is en.EfficientNet

    def test_construction_b2(self):
        """ Construction test of the B2 model.
        """
        model = en.efficientnet_factory(
            input_shape=[260, 260, 3],
            output_shape=[1000],
            id=en.VersionID.B2)
        assert type(model) is en.EfficientNet

    def test_construction_b3(self):
        """ Construction test of the B3 model.
        """
        model = en.efficientnet_factory(
            input_shape=[300, 300, 3],
            output_shape=[1000],
            id=en.VersionID.B3)
        assert type(model) is en.EfficientNet

    def test_construction_b4(self):
        """ Construction test of the B4 model.
        """
        model = en.efficientnet_factory(
            input_shape=[380, 380, 3],
            output_shape=[1000],
            id=en.VersionID.B4)
        assert type(model) is en.EfficientNet

    def test_construction_b5(self):
        """ Construction test of the B5 model.
        """
        model = en.efficientnet_factory(
            input_shape=[456, 456, 3],
            output_shape=[1000],
            id=en.VersionID.B5)
        assert type(model) is en.EfficientNet

    def test_construction_b6(self):
        """ Construction test of the B6 model.
        """
        model = en.efficientnet_factory(
            input_shape=[528, 528, 3],
            output_shape=[1000],
            id=en.VersionID.B6)
        assert type(model) is en.EfficientNet

    def test_construction_b7(self):
        """ Construction test of the B7 model.
        """
        model = en.efficientnet_factory(
            input_shape=[600, 600, 3],
            output_shape=[1000],
            id=en.VersionID.B7)
        assert type(model) is en.EfficientNet


if __name__ == '__main__':
    unittest.main()

