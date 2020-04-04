import unittest
import learn.models.dlibfacedetect as fd
from PIL import Image
import numpy as np
import learn.utils.utils as utils


model = fd.DlibHOGFaceDetector()
fd.crop_all_faces("/run/media/jay/4EFC223FFC2221A7/dataset", "/run/media/jay/4EFC223FFC2221A7/faces", model)
