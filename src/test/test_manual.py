
import learn.models.facerecogniton as fr


#r = rec.FaceRecognizer(config_file_path="../../res/models/ARCFACE_001/config/setup.yaml")
#r.train("/run/media/jay/4EFC223FFC2221A7/faces/")

#r = rec.FaceRecognizer(config_file_path="../../res/models/ARCFACE_001/config/setup.yaml")
#r.predict("ckpt-e09.hdf5", "/run/media/jay/4EFC223FFC2221A7/test")

f = fr.FaceRecognitionManager("/run/media/jay/4EFC223FFC2221A7/faces", "*.jpeg")
r = f.classify("/run/media/jay/4EFC223FFC2221A7/test/1940_2.jpeg")
print(r)
