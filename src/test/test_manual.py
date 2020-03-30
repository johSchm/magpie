#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:     For quick and dirty manual tests.
------------------------------------------- """


import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, \
    ReduceLROnPlateau, CSVLogger, LearningRateScheduler, TerminateOnNaN, LambdaCallback
import learn.models.arcface as af
import learn.models.scheduler.cosineanneal as ca


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--num-features', default=3, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--scheduler', default='CosineAnnealing',
                        choices=['CosineAnnealing', 'None'],
                        help='scheduler: ' +
                            ' | '.join(['CosineAnnealing', 'None']) +
                            ' (default: CosineAnnealing)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min-lr', default=1e-3, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.5, type=float)
    args = parser.parse_args()
    return args


args = parse_args()

# add model name to args
args.name = 'mnist_%s_%dd' %("arcface", args.num_features)

os.makedirs('models/%s' %args.name, exist_ok=True)

print('Config -----')
for arg in vars(args):
    print('%s: %s' %(arg, getattr(args, arg)))
print('------------')

joblib.dump(args, 'models/%s/args.pkl' %args.name)

with open('models/%s/args.txt' %args.name, 'w') as f:
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)), file=f)

(X, y), (X_test, y_test) = mnist.load_data()

X = X[:, :, :, np.newaxis].astype('float32') / 255
X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255

y = tf.keras.utils.to_categorical(y, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

if args.optimizer == 'SGD':
    optimizer = SGD(lr=args.lr, momentum=args.momentum)
elif args.optimizer == 'Adam':
    optimizer = Adam(lr=args.lr)

model = af.ArcFace([], [10])
model.print()

callbacks = [
    ModelCheckpoint(os.path.join('models', args.name, 'model.hdf5'),
        verbose=1, save_best_only=True),
    CSVLogger(os.path.join('models', args.name, 'log.csv')),
    TerminateOnNaN()]

if args.scheduler == 'CosineAnnealing':
    callbacks.append(ca.CosineAnnealingScheduler(
        T_max=args.epochs, eta_max=args.lr, eta_min=args.min_lr, verbose=1))

if True:#'face' in args.arch:
    # callbacks.append(LambdaCallback(on_batch_end=lambda batch, logs: print('W has nan value!!') if np.sum(np.isnan(model.layers[-4].get_weights()[0])) > 0 else 0))
    model.get_model().fit([X, y], y, validation_data=([X_test, y_test], y_test),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1)
else:
    model.fit(X, y, validation_data=(X_test, y_test),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1)

model.load_weights(os.path.join('models/%s/model.hdf5' %args.name))
if 'face' in args.arch:
    score = model.evaluate([X_test, y_test], y_test, verbose=1)
else:
    score = model.evaluate(X_test, y_test, verbose=1)

print("Test loss:", score[0])
print("Test accuracy:", score[1])