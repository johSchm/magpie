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

    def test_construction_values(self):
        """ Construction test of the model.
        """
        self.assertRaises(ValueError, en.EfficientNet())


if __name__ == '__main__':
    unittest.main()

