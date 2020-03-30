#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:     Test for ArcFace.
------------------------------------------- """


import unittest
import learn.models.arcface as af


class TestArcFaceNetPureTop(unittest.TestCase):
    """ Test for ArcFace without weights and with top included.
    """

    def test_construction_top(self):
        """ Construction test of the ArcFace model.
        """
        model = af.ArcFace(
            input_shape=[224, 224, 3],
            output_shape=[10],
            include_top=True)
        self.assertEqual(type(model), af.ArcFace)


if __name__ == '__main__':
    unittest.main()

