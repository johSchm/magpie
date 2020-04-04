import unittest
import learn.models.dlibfacedetect as fd
from PIL import Image
import numpy as np
import learn.utils.utils as utils


model = fd.DlibHOGFaceDetector()
img = Image.open("../../res/dataset/test_image_barack_obama.jpg")
pred = model.apply(np.array(img))
fd.crop_face(img, pred)
