#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:       https://github.com/4uiiurz1/keras-arcface
@todo:
@bug:
@brief:     The implementation of the cosine annealing scheduler.
------------------------------------------- """


import math
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        """ Init. method.
        :param T_max:
        :param eta_max:
        :param eta_min:
        :param verbose:
        """
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        """ At beginning of a new epoch.
        :param epoch:
        :param logs:
        :return:
        """
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        """ On the end of a epoch.
        :param epoch:
        :param logs:
        :return:
        """
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
