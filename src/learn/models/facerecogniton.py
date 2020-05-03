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


class FaceRecognitionManager:
    """ A wrapper for the face recognition library.
    """

    def __init__(self, data_path: str, file_encoding="*.jpg"):
        """ Init. method.
        :param file_encoding (str): The encoding of the image files.
        :param data_path (str): The path to the data.
        """
        self._file_encoding = file_encoding
        self._known_face_encodings = {}
        self._dirname = os.path.dirname(__file__)
        if data_path is not None:
            self._path = data_path
        else:
            self._path = os.path.join(self._dirname, 'known_people/')
        self._learn_faces()

    def _learn_faces(self):
        """ Goes through the data path images and learns the face specifics.
        """
        for folder in os.listdir(self._path):
            if os.path.isdir(os.path.join(self._path, folder)):
                path = os.path.join(self._path, folder, self._file_encoding)
                list_of_files = [f for f in glob.glob(path)]
                encodings = []
                for i, image_path in enumerate(list_of_files):
                    image = face_recognition.load_image_file(list_of_files[i])
                    try:
                        encodings.append(face_recognition.face_encodings(image)[0])
                    except IndexError:
                        pass
                    self._known_face_encodings[folder] = np.mean(encodings, axis=0)

    def get_known_names(self) -> list:
        """ Getter for all known face names.
        :return: (list) List of all known face names.
        """
        return list(self._known_face_encodings.keys())

    def get_known_encodings(self) -> list:
        """ Getter for all known face encodings.
        :return: (list) List of all known face encodings.
        """
        return list(self._known_face_encodings.values())

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
            matches = face_recognition.compare_faces(self.get_known_encodings(), face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.get_known_encodings(), face_encoding)
            best_match_index = int(np.argmin(face_distances))
            if matches[best_match_index]:
                name = self.get_known_names()
            face_names.append(name[best_match_index])
        return face_names
