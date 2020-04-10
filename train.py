#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:     Train the models.
------------------------------------------- """


from property import settingsManager as sManager
import property.property_utils as prop_utils
import learn.gpu.hvd_wrapper as hvd
import argparse

DEBUG_MODE = False
BASE_PATH = "../res/models/"

parser = argparse.ArgumentParser(description='Training.')
parser.add_argument('--sidx', type=int, help='Settings file Index.', default=0)

args = parser.parse_args()
settings_idx = args.sidx

master_file = prop_utils.get_settings_file_path()["settings_" + str(settings_idx)]
print("\n\nTraining: " + str(master_file) + "\n\n")
for model_config_idx in range(len(master_file)):
    print("Loading config: " + str(master_file[model_config_idx]))

    # load config parameters
    settings_manager = sManager.SettingsManager(settings_idx=settings_idx, model_idx=model_config_idx)
    input_shape = settings_manager.read("input_shape")
    checkpoint_path = settings_manager.read("checkpoint_path")
    validation_split = settings_manager.read("validation_split")
    batch_size = settings_manager.read("batch_size")
    epochs = settings_manager.read("epochs")
    data_path = settings_manager.read("data_path")
    steps_per_epoch = settings_manager.read("steps_per_epoch")
    class_path = settings_manager.read("class_path")
    model_type = settings_manager.read("model")
    seed = settings_manager.read("seed")
    data_identifier = settings_manager.read("data_identifier")
    gpu = settings_manager.read("gpu")
    weight_links = settings_manager.read("weights")
    hyper_param_path = settings_manager.read("hyper_parameter_path")
    key = settings_manager.read("key")
    log_path = settings_manager.read("log_path")
    model_path = settings_manager.read("model_path")
    model_format = settings_manager.read("model_format")

    if gpu == "parallel":
        hvd.init()
        parallel = True
    else:
        hvd.restrict_to_gpu(int(gpu))
        parallel = False

    # imports are moved here, due to GPU init
    from learn import learner as lrn
    from prepro import preprocessing as pro
    import data.data_utils as utils
    import parser.storage as store
    import data.sample as sample
    import os
    import learn.model.model_utils as model_utils
    import utils.path_utils as path_utils
    import utils.os_utils as os_utils
    import tensorflow as tf
    from keras import backend

    # force channels-last ordering
    tf.keras.backend.set_image_data_format('channels_last')
    backend.set_image_data_format('channels_last')
    print("Enforcing channel ordering: " + str(backend.image_data_format()))
    print("Enforcing channel ordering: " + str(tf.keras.backend.image_data_format()))

    # adapted relative paths
    hyper_param_path = path_utils.join(os.getcwd(), BASE_PATH, key, hyper_param_path)
    checkpoint_path = path_utils.join(os.getcwd(), BASE_PATH, key, checkpoint_path)
    class_path = path_utils.join(os.getcwd(), BASE_PATH, key, class_path)
    log_path = path_utils.join(os.getcwd(), BASE_PATH, key, log_path)
    model_path = path_utils.join(os.getcwd(), BASE_PATH, key, model_path, key + model_format)

    if os_utils.get_operation_system() == os_utils.OperatingSystems.WIN:
        checkpoint_path = checkpoint_path.replace('/', '\\')
        model_path = model_path.replace('/', '\\')
        log_path = log_path.replace('/', '\\')

    # setup model hyper parameter
    settings_manager = sManager.SettingsManager(path=hyper_param_path)
    loss = settings_manager.read("loss")
    optimizer = settings_manager.read("optimizer")
    metrics = settings_manager.read("metrics")

    LOSS = model_utils.get_loss_func(loss)
    OPTIMIZER = model_utils.init_optimizer(optimizer)
    METRICS = model_utils.get_metric_key(metrics)

    weight_links = model_utils.get_weight_dict(weight_links, checkpoint_path=checkpoint_path)

    processor = pro.Preprocessor(
        data_pattern=utils.DataPattern.TF_RECORD)
    if not DEBUG_MODE:
        train, val = store.FileManager().load(
            data_path,
            desired_shape={
                "a": input_shape["RGB"] if "RGB" in input_shape else None,
                "b": input_shape["optical_flow"] if "optical_flow" in input_shape else None,
                "c": input_shape["optical_flow_lk"] if "optical_flow_lk" in input_shape else None,
                "d": input_shape["optical_flow_tvl1"] if "optical_flow_tvl1" in input_shape else None,
                "e": input_shape["pose"] if "pose" in input_shape else None,
                "f": input_shape["segmentation"] if "segmentation" in input_shape else None},
            keypoints=False,
            data_identifier=data_identifier,
            return_type=utils.ReturnTypes.DATASET,
            sample_type=sample.SampleType.VIDEO)
    else:
        train, val = None, None
    learner = lrn.Learner(
        input_shape=input_shape,
        output_shape=prop_utils.get_num_of_items(class_path),
        model_id=model_type,
        seed=seed,
        log_path=log_path,
        ckpt_path=checkpoint_path,
        parallel=parallel,
        weight_links=weight_links,
        optimizer=OPTIMIZER,
        metrics=METRICS,
        loss=LOSS)
    learner.model.print()
    learner.save(model_path)
    if model_type == model_utils.ModelID.T3D or model_type == model_utils.ModelID.T3D.value:
        rnd_sample_frame = True
    else:
        rnd_sample_frame = False
    learner.train(
        train,
        validation_db=val,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        dataset_path=os.path.join(data_path, 'train'),
        rnd_sample_frame=rnd_sample_frame)
    learner.model.clear()

