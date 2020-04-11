# magpie

**:exclamation: DISCLAIMER: NOTE THIS TOOL IS CURRENTLY UNDER DEVELOPMENT AND 
A WORKING VERSION HAS NOT BEEN RELEASED YET :exclamation:**

No time to sort your personal pictures and even when you find some time,  
you merely sit in front of the screen contemplating a diversity of  
folder hierarchies, which could work for you?  
Then, let *magpie* simply do the job!

<!-- UPDATE via (cd in project dir) and $ doctoc . -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [System Requirements](#system-requirements)
- [Usage](#usage)
- [Implementation](#implementation)
  - [Image Classification](#image-classification)
  - [Face Recognition](#face-recognition)
    - [face_recognition](#face_recognition)
    - [ArcFace](#arcface)
- [Configurations](#configurations)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Future Development](#future-development)
- [References](#references)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## System Requirements
The following system configurations has been tested for deployment:

| OS                                | Python        |
| -------------                     |:-------------:|
| Arch Linux 5.4.26-1-lts           | 3.8.2 |

## Usage
For the framework can be employed, please make sure that
the [system requirements](#system-requirements) are satisfied.
That said, the tool can be executed and used by:
```
python magpie.py [-h] [-d] [-t] [-c]
```

| SHORT | LONG | DESCRIPTION |
| --- | --- | --- |
| -h | --help | Show this help message and exit.
| -d | --data_path | The path to the data.
| -t | --training | Initiate a new training phase.
| -c | --classify | Classifies the data under the provided data path.

## Implementation
This framework uses state-of-the-art machine learning tools to classify objects within images.

### Image Classification
One of those is the *EfficientNet* [[1]](#references), which reached state-of-the-art performances
on the popular image classification benchmark dataset *ImageNet* [[2, 3]](#references).
The implementation of the related model was highly inspired by [[4, 5]](#references).

### Face Recognition
The user can choose between the following face recognition methods:
Either the traditional, commonly used, and default classification via the
face_recognition [[10]](#references) library.
Or, alternatively, state-of-the-art *ArcFace* [[6]](#references) can be employed. 

#### face_recognition
The well known Python face_recognition [[10]](#references) library.
This method for face recognition is highly advisable for most of the users, preferring 
a more plug-and-play nature.

#### ArcFace
*ArcFace* [[6]](#references). is a novel approach for deep face recognition. This method reached state-of-the-art performances
on well-known face databases, like LFW [[7, 8]](#references). Note that, for the implementation, the code
from [[9]](#references) has been used. Note that this model comes untrained, thus a explicit training phase is 
inevitable, which undeniably require some time. 

## Configurations
All user-adjustable settings are located in [res](res).
The model settings can be found under [res/models](res/models).

## Deployment
(coming soon)

## Documentation
For a detailed code documentation, please refer to
[documentation](https://rawcdn.githack.com/johSchm/magpie/master/doc/_build/html/index.html).

## Troubleshooting
Some errors and the related fixes, encountered during development, are discussed [here](doc/readme/TROUBLESHOOTING.md).

## Future Development
So, what's up next? Open issues and hence possible future features are listed in the `Issue`
section of GitHub. If you like to participate in the development process, PRs are always
welcome. If you just like to recommend some features for future versions or indicate some 
yet unknown bugs, then just let me know.

## References
- [1] https://arxiv.org/abs/1905.11946
- [2] http://www.image-net.org/
- [3] https://paperswithcode.com/sota/image-classification-on-imagenet
- [4] https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- [5] https://github.com/titu1994/keras-efficientnets
- [6] https://arxiv.org/abs/1801.07698
- [7] http://vis-www.cs.umass.edu/lfw/
- [8] https://paperswithcode.com/sota/face-verification-on-labeled-faces-in-the
- [9] https://github.com/4uiiurz1/keras-arcface
- [10] https://pypi.org/project/face-recognition/