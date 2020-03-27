#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       2020
refs:
todos:
------------------------------------------- """


from random import randint
import data.image.image_utils as image_utils
import numpy as np
import tensorflow as tf
import property.property_utils as prop_utils
import live.feeder as feed
import sensors.sensor_utils as sensor_utils
import sensors.kinect.bridge as kinect
import time
import _thread
import keyboard
import cv2
import learn.prediction.visu.visualizer as visu
import matplotlib.pyplot as plt
import psutil
from gpuinfo import GPUInfo
import matplotlib.ticker as ticker
import utils.path_utils as path_utils


PREDICTION_DELAY = 0.5  # s
RECORDING_DELAY = 0.1  # s
HW_LOG_DELAY = 0.1  # s

_glob_online_frames = None
_glob_online_timestamp = time.time()
_glob_last_pred = None
_glob_prediction_time_tracking = []
_glob_stop_flag = False


class PredictionController:
    """ Handles online and offline predictions.
    """

    def __init__(self, model, class_path, quiet=True, visu=True,
                 optical_flow_estimator=None, log_path=None):
        """ Init. method.
        :param optical_flow_estimator:
        :param model:
        :param class_path:
        :param quiet:
        :param visu:
        :param log_path:
        """
        self._optical_flow_estimator = optical_flow_estimator
        self._quiet = quiet
        self._visu = visu
        self._model = model
        self._model.summary()
        try:
            self._input_shape = self._model.get_layer('input').input.shape[1:]
            self._video_length = self._input_shape[0]
        except ValueError:
            self._input_shape = [
                self._model.get_layer('2d_input').input.shape[1:],
                self._model.get_layer('3d_input').input.shape[1:]]
            self._video_length = 24#self._input_shape[0][0]
        self._classes = prop_utils.load_json(class_path)
        self._feeder = None
        self._log_path = log_path

    def predict_offline(self, sample_batch, raw=False):
        """ Predicts the labels of a batch of samples.
        :param sample_batch:
        :param raw: returns raw labels (not mapped)
        :return: labels
        """
        global _glob_last_pred, _glob_prediction_time_tracking
        start_time = time.time()
        if type(sample_batch) is not list and \
                type(sample_batch) is not np.ndarray and \
                type(sample_batch) is not tf.Tensor:
            return []
        if type(sample_batch[0]) is str:
            sample_batch = image_utils.load_images_from_paths(
                sample_batch, self._input_shape)
        if type(sample_batch[0]) is dict:
            sample_batch = [[item, ] for item in sample_batch[0].values()]
        if type(sample_batch) is not tf.Tensor:
            if type(self._input_shape) is list:
                if self._input_shape[1][-1] == 2:
                    sample_batch[1] = np.array(sample_batch[1])[:, :, :, :, :-1]
                    sample_batch = [
                        tf.convert_to_tensor(sample_batch[0], dtype=tf.float32),
                        tf.convert_to_tensor(sample_batch[1], dtype=tf.float32)]
                    sample_batch = tf.convert_to_tensor(sample_batch, dtype=tf.float32)
                elif self._input_shape[0] == self._input_shape[1][1:]:
                    sample_batch[1] = np.array(sample_batch[0])[:, randint(0, 23)]
                    sample_batch = [
                        tf.convert_to_tensor(sample_batch[1], dtype=tf.float32),
                        tf.convert_to_tensor(sample_batch[0], dtype=tf.float32)]
            else:
                sample_batch = tf.convert_to_tensor(sample_batch, dtype=tf.float32)
        print("Input Shape: " + str(sample_batch.shape))
        result = self._model.predict_on_batch(sample_batch)
        if not raw:
            result = self._improve_prediction(result)
        if not self._quiet:
            for r, res in enumerate(result):
                self._print_result(r, res)
        #if result != _glob_last_pred:
        #    print(">> PREDICTION CHANGED!!")
        _glob_last_pred = result
        end_time = time.time()
        _glob_prediction_time_tracking.append(end_time - start_time)
        print("Time elapsed for prediction: " + str(round(end_time - start_time, 3)) + "s")
        return result

    def predict_online(self):
        """ Uses the models and a sample feeder to predict samples
        in an online fashion.
        """
        shape = [self._input_shape.dims[i].value for i in range(len(self._input_shape.dims))]
        feeder = feed.ModelFeeder(
            optical_flow_estimator=self._optical_flow_estimator,
            video_length=self._video_length,
            storage_path=None,
            callback=self._predict_cb,
            sensor_type=sensor_utils.SensorType.KINECT,
            data_formats=[kinect.KinectFrameFormats.COLOR],
            display=True,
            display_cb=self._visu_cb,
            sleep=RECORDING_DELAY,
            shape=np.array(shape)[1:])
        self._feeder = feeder
        _thread.start_new_thread(self._run_online_prediction, ())
        if self._log_path is not None:
            _thread.start_new_thread(self._track_hw_usage, ())
        feeder.run()

    @staticmethod
    def _visu_cb(frame):
        """ Callback for visualization.
        :param frame:
        """
        global _glob_last_pred
        if _glob_last_pred is not None:
            pos_x, pos_y = 10, 750
            max_class = visu.get_best_class(_glob_last_pred[0])
            for elem in sorted(_glob_last_pred[0].items(), reverse=True):
                txt = "{0}: {1}%".format(elem[0], elem[1])
                stroke = 10 if elem[0] == max_class else 5
                size = 2.5 if elem[0] == max_class else 1.5
                frame["RGB"] = visu.txt_in_img(
                    frame["RGB"], txt, color=(30, 30, 200, 255),
                    size=size, stroke=stroke, position=(pos_x, pos_y))
                pos_y += 60
        if "optical_flow" in frame and frame["optical_flow"] is not None:
            image_utils.show(cv2.resize(frame["optical_flow"], (540, 360)), 'OpticalFlow', True)
        image_utils.show(cv2.resize(frame["RGB"], (1080, 720)), 'RGB', True)
        return

    @staticmethod
    def _predict_cb(frames):
        """ Callback method for online prediction.
        :param frames:
        """
        global _glob_online_frames, _glob_online_timestamp
        _glob_online_frames = frames
        _glob_online_timestamp = time.time()

    def _run_online_prediction(self):
        """ Starts the online prediction process and output.
        """
        global _glob_online_frames, _glob_online_timestamp, _glob_stop_flag
        previous_timestamp = 0
        while True:
            if _glob_online_frames is not None \
                    and (_glob_online_timestamp != previous_timestamp or previous_timestamp == 0):
                print("predicting ...")
                if type(self._input_shape) is not list:
                    self.predict_offline([_glob_online_frames["RGB"], ], raw=False)
                else:
                    self.predict_offline([_glob_online_frames, ], raw=False)
                previous_timestamp = _glob_online_timestamp
                time.sleep(PREDICTION_DELAY)
            if keyboard.is_pressed('q') or _glob_stop_flag:
                print("\nExiting Online Prediction ...")
                _glob_stop_flag = True
                break
            time.sleep(1)
        self._feeder.close()
        return

    def _plot_time(self, plot=False):
        """ Plots stats gathered during prediction
        :param plot:
        """
        global _glob_prediction_time_tracking
        if len(_glob_prediction_time_tracking) > 1:
            run_count = np.arange(0, len(_glob_prediction_time_tracking), 1)
            if plot:
                fig, ax = plt.subplots()
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                plt.plot(run_count, _glob_prediction_time_tracking, color='royalblue')
                plt.xlabel('Prediction Count')
                plt.ylabel('Time [s]')
                plt.title('Prediction Time')
                plt.savefig(path_utils.join(self._log_path, "prediction_time.png"), format='png')
                plt.clf()
                plt.close()
            else:
                np.save(path_utils.join(self._log_path, "prediction_time.npy"),
                        np.array([run_count, _glob_prediction_time_tracking]))

    def _track_hw_usage(self):
        """ Tracks the hardware usage.
        {
            'CPU': 93.8,                    -> CPU usage
            'RAM': {
                'total': 8250470400,
                'available': 2417065984,
                'percent': 70.7,            -> RAM usage
                'used': 5833404416,
                'free': 2417065984
            },
            'GPU': (
                [6],                        -> usage percent
                [4010]                      -> memory used
            )
        }
        """
        usage = {
            "TIME": [],
            "CPU": [],
            "GPU": [],
            "RAM": []
        }
        start_time = time.time()
        while True:
            global _glob_stop_flag
            self._plot_time()
            cpu = psutil.cpu_percent()
            ram = dict(psutil.virtual_memory()._asdict())
            gpu = GPUInfo.gpu_usage()
            usage["TIME"].append(round(time.time() - start_time, 2))
            usage["CPU"].append(cpu)
            usage["GPU"].append(list(gpu)[0][0])
            usage["RAM"].append(ram["percent"])
            self._plot_hw_usage(usage)
            time.sleep(HW_LOG_DELAY)
            if keyboard.is_pressed('q') or _glob_stop_flag:
                print("\nExiting HW tracking ...")
                _glob_stop_flag = True
                break

    def _plot_hw_usage(self, usage, plot=False):
        """ Plots the HW usage.
        :param usage
        :param plot
        """
        if plot:
            plt.plot(usage["TIME"], usage["CPU"], color='royalblue', label="CPU")
            plt.plot(usage["TIME"], usage["GPU"], color='teal', label="GPU")
            plt.plot(usage["TIME"], usage["RAM"], color='dimgray', label="RAM")
            plt.xlabel('Time [s]')
            plt.ylabel('Usage [%]')
            plt.title('Hardware Usage')
            plt.legend(loc='center right')
            plt.savefig(path_utils.join(self._log_path, "hw_usage_log.png"), format='png')
            plt.clf()
            plt.close()
        else:
            np.save(path_utils.join(self._log_path, "hw_usage_log.npy"),
                    np.array([usage["TIME"], usage["CPU"], usage["GPU"], usage["RAM"]]))

    def _improve_prediction(self, raw_results):
        """ Improves the label style.
        :param raw_results:
        :return: dict of label name and prediction value
        """
        raw_results = raw_results.numpy()
        improved_results = []
        for r, result in enumerate(raw_results):
            improved_result = {}
            for p, value in enumerate(result):
                prediction_percentage = round(value * 100, 3)
                label_name = self._map_label_idx_to_name(p)
                improved_result[label_name] = prediction_percentage
            improved_results.append(improved_result)
        return improved_results

    def _map_label_idx_to_name(self, label_idx):
        """ Maps labels indexes to names
        :param label_idx:
        :return: names
        """
        label_name = self._classes[str(label_idx)]
        return label_name

    @staticmethod
    def _print_result(sample_id, prediction):
        """ Prints labels in console.
        :param sample_id
        :param prediction:
        """
        if type(prediction) is dict:
            print("\n--> Prediction for Sample {0}:".format(sample_id))
            for class_name in prediction.keys():
                print("\t{0}: {1}%".format(class_name, prediction[class_name]))
        else:
            print("--> Image Sequence: " + str(sample_id) + " | Prediction: " + str(prediction))
