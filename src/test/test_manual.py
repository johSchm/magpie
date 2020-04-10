import unittest
import learn.models.dlibfacedetect as fd
from PIL import Image
import numpy as np
import learn.utils.utils as utils
import core.face.recognizer as rec


r = rec.FaceRecognizer(config_file_path="../../res/models/ARCFACE_001/config/setup.yaml")
r.train("/run/media/jay/4EFC223FFC2221A7/faces/")
