#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:     Test for the face recognition manager object.
------------------------------------------- """


import unittest
import learn.models.facerecogniton as fr


class TestFaceRecognitionManager(unittest.TestCase):
    """ Test for the face recognition manager object.
    """

    def test_number_of_encodings(self):
        """ Initiates the face recognition manager and test if it can learn the test faces for:
            [Obama, Trump, Bush]
        """
        frm = fr.FaceRecognitionManager(data_path="../../res/dataset/test")
        with self.subTest():
            self.assertEqual(type(frm), fr.FaceRecognitionManager)
        with self.subTest():
            encodings = frm.get_known_encodings()
            self.assertEqual(len(encodings), 3)

    def test_known_faces(self):
        """ Initiates the face recognition manager and test if it can learn the test faces for:
            [Obama, Trump, Bush]
        """
        frm = fr.FaceRecognitionManager(data_path="../../res/dataset/test")
        names = frm.get_known_names()
        with self.subTest():
            self.assertEqual(len(names), 3)
        with self.subTest():
            self.assertTrue("obama" in names)
        with self.subTest():
            self.assertTrue("trump" in names)
        with self.subTest():
            self.assertTrue("bush" in names)

    def test_face_recognition(self):
        """ Initiates the face recognition manager and test if it can learn the test faces for:
            [Obama, Trump, Bush]
        """
        frm = fr.FaceRecognitionManager(data_path="../../res/dataset/test")
        with self.subTest():
            prediction = frm.classify(image_path="../../res/dataset/test/obama/obama_000.jpg")
            self.assertEqual(prediction[0], "obama")
        with self.subTest():
            prediction = frm.classify(image_path="../../res/dataset/test/trump/trump_000.jpg")
            self.assertEqual(prediction[0], "trump")
        with self.subTest():
            prediction = frm.classify(image_path="../../res/dataset/test/bush/bush_000.jpg")
            self.assertEqual(prediction[0], "bush")


if __name__ == '__main__':
    unittest.main()

