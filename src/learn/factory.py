#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:
todo:
------------------------------------------- """


import learn.models.image.bic as bic
import learn.models.video.bvc as bvc
import learn.models.video.mcvc as mcvc
import learn.models.video.c3d as c3d
import learn.models.video.td_vgg as td_vgg
import data.sample as sample
import learn.utils.utils as utils
import learn.models.video.ts2d as two_stream
import learn.models.video.i3d as i3d
import learn.models.video.stlstm as stlstm
import learn.models.video.i3d_2s as i3d_2s
import learn.models.video.t3d as t3d


def model_factory(input_shape, model_id, output_shape=None, 
                  parallel=False, weight_links=None, optimizer=None,
                  metrics=None, loss=None, log_path=None,
                  ckpt_path=None):
    """ A factory method for models.
    :param input_shape:
    :param output_shape:
    :param weight_links:
    :param model_id:
    :param parallel:
    :param optimizer:
    :param metrics:
    :param loss:
    :param log_path:
    :param ckpt_path:
    :return: models.
    """
    if model_id == utils.ModelID.BIC \
            or model_id == utils.ModelID.BIC.value:
        return bic.BICModel(
            input_shape=input_shape, parallel=parallel,
            optimizer=optimizer, metrics=metrics, loss=loss,
            log_path=log_path, ckpt_path=ckpt_path)
    elif model_id == utils.ModelID.BVC \
            or model_id == utils.ModelID.BVC.value:
        return bvc.BVCModel(
            input_shape=input_shape, parallel=parallel,
            optimizer=optimizer, metrics=metrics, loss=loss,
            log_path=log_path, ckpt_path=ckpt_path)
    elif model_id == utils.ModelID.MCVC \
            or model_id == utils.ModelID.MCVC.value:
        return mcvc.MCVCModel(
            input_shape=input_shape, parallel=parallel,
            optimizer=optimizer, metrics=metrics, loss=loss,
            log_path=log_path, ckpt_path=ckpt_path)
    elif model_id == utils.ModelID.C3D \
            or model_id == utils.ModelID.C3D.value:
        return c3d.C3DModel(
            input_shape=input_shape, output_shape=output_shape,
            parallel=parallel, log_path=log_path,
            weight_links=weight_links, ckpt_path=ckpt_path,
            optimizer=optimizer, metrics=metrics, loss=loss)
    elif model_id == utils.ModelID.TD_VGG \
            or model_id == utils.ModelID.TD_VGG.value:
        return td_vgg.TDVGGModel(
            input_shape=input_shape, output_shape=output_shape,
            parallel=parallel, log_path=log_path, ckpt_path=ckpt_path,
            optimizer=optimizer, metrics=metrics, loss=loss)
    elif model_id == utils.ModelID.TS2D \
            or model_id == utils.ModelID.TS2D.value:
        return two_stream.TwoStreamModel(
            input_shape=input_shape, output_shape=output_shape,
            parallel=parallel, log_path=log_path, ckpt_path=ckpt_path,
            optimizer=optimizer, metrics=metrics, loss=loss)
    elif model_id == utils.ModelID.I3D \
            or model_id == utils.ModelID.I3D.value:
        return i3d.I3DModel(
            input_shape=input_shape, output_shape=output_shape,
            parallel=parallel, log_path=log_path, ckpt_path=ckpt_path,
            optimizer=optimizer, metrics=metrics, loss=loss,
            weight_links=weight_links)
    elif model_id == utils.ModelID.I3D2S \
            or model_id == utils.ModelID.I3D2S.value:
        return i3d_2s.I3D2StreamModel(
            input_shape=input_shape, output_shape=output_shape,
            parallel=parallel, log_path=log_path, ckpt_path=ckpt_path,
            optimizer=optimizer, metrics=metrics, loss=loss,
            weight_links=weight_links)
    elif model_id == utils.ModelID.STLSTM \
            or model_id == utils.ModelID.STLSTM.value:
        return stlstm.STLSTMModel(
            input_shape=input_shape, output_shape=output_shape,
            parallel=parallel, log_path=log_path, ckpt_path=ckpt_path,
            optimizer=optimizer, metrics=metrics, loss=loss)
    elif model_id == utils.ModelID.T3D \
            or model_id == utils.ModelID.T3D.value:
        return t3d.T3DModel(
            input_shape=input_shape, output_shape=output_shape,
            parallel=parallel, log_path=log_path, ckpt_path=ckpt_path,
            optimizer=optimizer, metrics=metrics, loss=loss)
    else:
        raise NotImplementedError("Other sample types are not yet implemented!")


def model_factory_auto(
        input_shape, output_shape=None,
        sample_type=sample.SampleType.IMAGE,
        *args):
    """ A factory method for models.
    This will automatically search for the best fitting models
    based on the passed parameters.
    :param input_shape:
    :param output_shape:
    :param args:
    :param sample_type:
    :return: proper models.
    """
    if sample_type == sample.SampleType.IMAGE:
        return bic.BICModel(
            input_shape=input_shape, *args)
    elif sample_type == sample.SampleType.VIDEO:
        return bvc.BVCModel(
            input_shape=input_shape, *args)
    else:
        raise NotImplementedError("Other sample types are not yet implemented!")
