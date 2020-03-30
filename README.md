# magpie

No time to sort your personal pictures and even when you find some time,  
you merely sit in front of the screen contemplating a diversity of  
folder hierarchies, which could work for you?  
Then, let *magpie* simply do the job!

<!-- UPDATE via (cd in project dir) and $ doctoc . -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [System Requirements](#system-requirements)
- [Implementation](#implementation)
- [Configurations](#configurations)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Future Development](#future-development)
- [References](#references)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## System Requirements
The following system configurations has been tested for deployment:

| OS                                | Python        |
| -------------                     |:-------------:|
| Arch Linux 5.4.26-1-lts           | 3.8.2 |

## Implementation
This framework uses state-of-the-art machine learning tools to classify objects within images.
One of those is the EfficientNet [[1]](#references), which reached state-of-the-art performances
on the popular image classification benchmark dataset ImageNet [[2, 3]](#references).
The implementation of the related model was highly inspired by [[4, 5]](#references).

## Configurations
All user-adjustable settings are located in [res](res).
The model settings can be found under [res/models](res/models).

## Deployment
(coming soon)

## Documentation
For a detailed code documentation, please refer to
[documentation](https://rawcdn.githack.com/johSchm/magpie/master/doc/_build/html/index.html).

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
