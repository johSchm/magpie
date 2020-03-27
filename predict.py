#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
------------------------------------------- """


import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)


from learn import learner as learn
from property import settingsManager as sManager
import utils.path_utils as path_utils
import cv2
import os
import numpy as np
import utils.array_utils as array_utils
import learn.prediction.prediction as predict
import prepro.optflow.optflow as optflow


BASE_PATH = "../res/models"
PREDICTION_PATHS = [
    #r"C:\Users\schmidt\PycharmProjects\test\walk\801",
    #r"C:\Users\schmidt\PycharmProjects\test\pick\221",
    #r"C:\Users\schmidt\PycharmProjects\test\carry\1809",
    #r"E:\IFF_HAR_DS\Technikum\carry\0",
    #r"E:\IFF_HAR_DS\Technikum\pick\0",
    #r"E:\IFF_HAR_DS\Technikum\walk\0",
    r"E:\IFF_HAR_DS\simone\carry\0",
    r"E:\IFF_HAR_DS\simone\pick\0",
    r"E:\IFF_HAR_DS\simone\walk\0"
]
CHECKPOINT = "ckpt-e05.hdf5"


settings_manager = sManager.SettingsManager(model_idx=0, settings_idx=23)
image_size = settings_manager.read("image_size")
checkpoint_path = settings_manager.read("checkpoint_path")
model_path = settings_manager.read("model_path")
key = settings_manager.read("key")
shape = settings_manager.read("input_shape")
class_path = settings_manager.read("class_path")
model_format = settings_manager.read("model_format")
optical_flow = settings_manager.read("optical_flow")
feature_detection = settings_manager.read("feature_detection")
log_path = settings_manager.read("log_path")

checkpoint_path = path_utils.join(BASE_PATH, key, checkpoint_path, CHECKPOINT)
model_path = path_utils.join(BASE_PATH, key, model_path, key + model_format)
class_path = path_utils.join(BASE_PATH, key, class_path)
log_path = path_utils.join(BASE_PATH, key, log_path)

learner = learn.Learner()
learner.load(model_path)
learner.model.get_model().load_weights(checkpoint_path)

optical_flow_estimator = optflow.OpticalFlowEstimator(
    estimation_method=optical_flow,
    feature_detection=feature_detection)
pc = predict.PredictionController(
    learner.model.get_model(), class_path, log_path=log_path,
    optical_flow_estimator=optical_flow_estimator)
pc.predict_online()
#pc.predict_offline(PREDICTION_PATHS)
