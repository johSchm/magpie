#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:       https://towardsdatascience.com/a-beginners-guide-to-building-your-own-face-recognition-system-to-creep-out-your-friends-df3f4c471d55
@todo:
@bug:
@brief:      A wrapper for the face recognition library.
------------------------------------------- """


import face_recognition
import cv2
import numpy as np
import os
import glob


class FaceRecognitionWrapper:
    """ A wrapper for the face recognition library.
    """

    def __init__(self, data_path: str, image_encoding="*.jpg"):
        """ Init. method.
        :param image_encoding (str): The encoding of the images.
        :param data_path (str): The path to the data.
        """
        self._encoding = image_encoding
        self._known_face_encodings = []
        self._known_face_names = []
        self._dirname = os.path.dirname(__file__)
        if data_path is not None:
            self._path = data_path
        else:
            self._path = os.path.join(self._dirname, 'known_people/')
        self._learn_faces()

    def _learn_faces(self):
        """ Goes through the data path images and learns the face specifics.
        """
        list_of_files = [f for f in glob.glob(self._path + self._encoding)]
        number_files = len(list_of_files)
        names = list_of_files.copy()
        for i in range(number_files):
            globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
            globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(
                globals()['image_{}'.format(i)])[0]
            self._known_face_encodings.append(globals()['image_encoding_{}'.format(i)])
            names[i] = names[i].replace("known_people/", "")
            self._known_face_names.append(names[i])

    def classify(self, image_path: str) -> list:
        """ Classifies the image and returns a list of all known persons involved.
        :param image_path: (str) The path to the image.
        :return: (list) A list of predictions.
        :raises ValueError if the provided image path is not a string.
                FileNotFoundError if the provided image path does not exist.
        """
        if type(image_path) is not str:
            raise ValueError("The image path needs to be a string, got instead {}".format(type(image_path)))
        if not os.path.exists(image_path):
            raise FileNotFoundError("The image under {} could not be found!".format(image_path))
        image = cv2.imread(image_path)
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        image = image[:, :, ::-1]
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self._known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self._known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self._known_face_names[best_match_index]
            face_names.append(name)
        return face_names
