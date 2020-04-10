#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
todo:
------------------------------------------- """


import property.settingsManager as sManager
import prepro.preprocessing as pro
import data.data_utils as data_utils
import data.sample as sample
import parser.storage as store
import utils.path_utils as path_utils


BASE_PATH = "../res/models/"


settings_manager = sManager.SettingsManager()
raw_data_path = settings_manager.read("raw_data_path")
data_path = settings_manager.read("data_path")
image_size = settings_manager.read("image_size")
partial_load = settings_manager.read("partial_load")
class_path = settings_manager.read("class_path")
test_split = settings_manager.read("test_split")
fpv = settings_manager.read("fpv")
augmentation_path = settings_manager.read("augmentation_path")
data_cache_path = settings_manager.read("data_cache_path")
from_cache = settings_manager.read("from_cache")
opt_flow = settings_manager.read("optical_flow")
segmentation = settings_manager.read("segmentation")
feature_detection = settings_manager.read("feature_detection")
pose_method = settings_manager.read("pose")
colormode = settings_manager.read("color_depth")
memory_data_storage = settings_manager.read("memory_data_storage")
key = settings_manager.read("key")

class_path = path_utils.join(BASE_PATH, key, class_path)
raw_data_path = path_utils.join(BASE_PATH, key, raw_data_path)
data_path = path_utils.join(BASE_PATH, key, data_path)
augmentation_path = path_utils.join(BASE_PATH, key, augmentation_path)
data_cache_path = path_utils.join(BASE_PATH, key, data_cache_path)

processor = pro.Preprocessor(
    colormode=colormode,
    data_pattern=data_utils.DataPattern.TF_RECORD,
    sample_type=sample.SampleType.VIDEO,
    memory_data_storage=memory_data_storage)
sample_train, sample_test = processor.run(
    raw_data_path,
    class_path,
    data_cache_path,
    image_size,
    test_split,
    partial_load,
    fpv,
    augmentation_file=augmentation_path,
    create_images=not from_cache,
    opt_flow_method=opt_flow,
    feature_detection=feature_detection,
    segmentation_method=segmentation,
    pose_method=pose_method)
store.FileManager().save(data_path, [sample_train, sample_test])
settings_manager.write("input_shape", processor.get_shape(), force=False, overwrite=False)
