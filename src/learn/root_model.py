#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:
todo:
------------------------------------------- """

from random import randint
import pandas as pd
import seaborn as sns
import property.property_utils as prop_utils
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.keras import backend
from tensorflow.compat.v1 import global_variables
from tensorflow.python.platform import gfile
from tensorflow.compat.v1 import GraphDef
from tensorflow.compat.v1 import Session
from tensorflow.keras.layers import BatchNormalization
import os
import learn.utils.utils as utils
import learn.log as log
import utils.path_utils as path_utils
import utils.data_utils as data_utils
import matplotlib.pyplot as plt
from PIL import Image
import learn.gpu.hvd_wrapper as hvd


def dummy_model():
    """ Returns a dummy models.
    :return: dummy models
    """
    return Model()


class Model:
    """ Model.
    """

    def __init__(self, input_shape=None, output_shape=None,
                 base_model=None, parallel=False,
                 log_path=None, ckpt_path=None):
        """ Init. Method.
        :param input_shape:s
        :param output_shape:
        :param base_model:
        :param parallel:
        :param log_path:
        :param ckpt_path:
        """
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._callbacks = log.setup_callbacks(log_path, ckpt_path)
        self._model = base_model
        self._num_classes = output_shape
        self._parallel = parallel

    def _configure(self, optimizer, loss, metrics):
        """
        Configures the models.
        :param optimizer:
        :param loss:
        :param metrics:
        """
        if self._parallel:
            optimizer = hvd.wrap_optimizer(optimizer)
        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metrics],
            experimental_run_tf_function=False)

    def _load_weights(self, weight_links):
        """ Loads pre-trained weights from the path.
        Args:
            weight_links (dict):
                {
                    path (str): Path to weight file.
                    http (str): Online link to weights for download.
                }
        """
        if not os.path.isfile(weight_links['path']):
            utils.download_weights(weight_links)
        self._model.load_weights(weight_links['path'])

    @staticmethod
    def get_norm_layer(norm, previous_layer=None):
        """ Returns the normlaization layer.
        :param norm:
        :param previous_layer:
        :return: layer
        """
        layer = None
        if norm == utils.Normalizations.BATCH_NORM or norm == utils.Normalizations.BATCH_NORM:
            layer = BatchNormalization(
                axis=-1, momentum=0.99, epsilon=0.001,
                center=True, scale=True, beta_initializer='zeros',
                gamma_initializer='ones', moving_mean_initializer='zeros',
                moving_variance_initializer='ones', beta_regularizer=None,
                gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
        if layer is None:
            return previous_layer
        return layer(previous_layer)

    def clear(self):
        """ Empties the models reference.
        """
        self._model = None

    def get_model(self):
        """ Returns the models.
        :return: models
        """
        return self._model

    def set_model(self, new_model):
        """ Sets the models to a new one.
        :param: new_model:
        """
        self._model = new_model

    def get_output_shape(self):
        """ Returns the output shape.
        :return: shape
        """
        return self._output_shape

    def get_input_shape(self, data_type):
        """ Returns the input shape.
        :return: shape
        """
        return self._input_shape[data_type]

    def save(self, path, protocol_buffer=False):
        """ Saves the learn.
        :param path
        :param protocol_buffer
        """
        if self._model is not None and path is not None:
            if protocol_buffer:
                frozen_graph = self._freeze_session(
                    backend.get_session(), output_names=[out.op.name for out in self.model.outputs])
                tf.io.write_graph(frozen_graph, path, 'learn', as_text=False)
            else:
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                self._model.save(path, overwrite=True)

    @staticmethod
    def _freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        """ Freezes a tensorflow session.
        :param session:
        :param keep_var_names:
        :param output_names:
        :param clear_devices:
        :return: frozen graph
        """
        if session is None:
            raise TypeError("Session is None!")
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

    def load(self, path, protocol_buffer=False):
        """ Saves the learn.
        :param path
        :param protocol_buffer
        :return protocol_buffer = False: learn
                protocol_buffer = True:  graph
        """
        if protocol_buffer:
            with Session() as sess:
                with gfile.FastGFile(os.path.join(path, 'learn'), 'rb') as file:
                    graph_def = GraphDef()
                    graph_def.ParseFromString(file.read())
                    sess.graph.as_default()
                    return tf.import_graph_def(graph_def)
        else:
            self._model = tf.keras.models.load_model(path)
            return self._model

    def train(self, dataset, validation_db=None,
              epochs=3, batch_size=32, validation_split=0.3,
              steps_per_epoch=None, dataset_path=None, rnd_sample_frame=False):
        """ Start training phase.
        :param dataset: TF database
        :param validation_db:
        :param epochs: number of epochs
        :param batch_size
        :param validation_split
        :param steps_per_epoch
        :param dataset_path:
        .param rnd_sample_frame:
        """
        if steps_per_epoch is None:
            if dataset_path is not None:
                steps_per_epoch = self._auto_compute_spe(ds_path=dataset_path, batch_size=batch_size)
            else:
                steps_per_epoch = self._auto_compute_spe(dataset=dataset, batch_size=batch_size)
        if self._parallel:
            dataset = hvd.wrap_dataset(dataset)
            epochs = hvd.wrap_epochs(epochs)
            if validation_db is not None:
                validation_db = hvd.wrap_dataset(validation_db)
        dataset = dataset.batch(batch_size)#.shuffle(32)
        if validation_db is not None:
            validation_db = validation_db.batch(batch_size)#.shuffle(32)
        generator = self.generator_batch(
            dataset, channel_first=False, use_paths=False, rnd_sample_frame=rnd_sample_frame)
        val_gen = self.generator_batch(
            validation_db, channel_first=False, use_paths=False, rnd_sample_frame=rnd_sample_frame)
        d = next(generator)
        e = next(val_gen)
        try:
            print("Training input shape:\t" + str(d[0].shape) + " and " + str(d[1].shape))
            print("Validation input shape:\t" + str(e[0].shape) + " and " + str(e[1].shape))
        except:
            print("Unable to show shape ...")
        #image_utils.show(d[0][0][12].numpy(), cv=False, title=str(d[1][0].numpy()))
        #image_utils.show(d[0][1][12].numpy(), cv=False, title=str(d[1][1].numpy()))
        self._model.fit(
            generator, epochs=epochs,
            steps_per_epoch=steps_per_epoch, callbacks=self._callbacks,
            validation_data=val_gen, validation_steps=10)

    @staticmethod
    def _auto_compute_spe(batch_size, dataset=None, ds_path=None):
        """ Computes the best number of steps per epoch.
        :param dataset:
        :param batch_size:
        :param ds_path:
        :return: spe
        """
        if dataset is not None:
            ds_length = data_utils.get_ds_length(
                dataset=dataset)
        elif ds_path is not None:
            ds_length = data_utils.get_ds_length(
                path=ds_path,
                path_pattern='*a.tfrecords')
        else:
            raise ValueError("Neither a dataset nor a path is defined!")
        steps_per_epoch = ds_length / batch_size
        return int(steps_per_epoch)

    @staticmethod
    def generator_batch(db, channel_first=False, use_paths=False, rnd_sample_frame=False):
        """ A generator for datasets.
        :param db:
        :param channel_first:
        :param use_paths:
        :param rnd_sample_frame:
        :return (yield) data
        """
        while True:
            for data in db:
                if type(data) is tuple:
                    if channel_first:
                        yield [np.moveaxis(item[0], item[0].ndim - 1, 1) for item in data][0], data[0][1]
                    elif use_paths:
                        imgs = []
                        # @todo @HACK dirty (hard coded shape)
                        for path in data[0][0]:
                            path = path.numpy().decode('utf-8')
                            paths = [path_utils.join(path, str(i) + '.jpg') for i in range(24)]
                            img = [np.array(Image.open(p).resize((224, 224))) / 255.0 for p in paths]
                            img = np.array(img).reshape([24, 224, 224, 3])
                            imgs.append(img)
                        imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
                        yield imgs, data[0][1]
                    else:
                        if len(list(data)) > 1:
                            imgs = [item[0] for item in data]
                        else:
                            imgs = [item[0] for item in data][0]
                        labels = data[0][1]
                        if rnd_sample_frame:
                            img = imgs[:, randint(0, 23)]
                            yield ([img, imgs], labels)
                        else:
                            yield imgs, labels
                else:
                    yield data

    @staticmethod
    def generator(db):
        """ A generator for datasets.
        :param (yield) data
        """
        while True:
            for data in db:
                val = tf.expand_dims(data[0], axis=0)
                label = data[1]
                yield val, label

    def evaluate(self, test, ds_path, class_path, log_path, batch_size=1):
        """ Evaluates the learn.
        :param test
        :param log_path:
        :param ds_path:
        :param batch_size:
        :param class_path:
        :return: results
        """
        test = test.batch(batch_size)
        generator = self.generator_batch(test, rnd_sample_frame=False)  # enable for T3D
        nb_validation_samples = self._auto_compute_spe(ds_path=ds_path, batch_size=batch_size)
        print("Test dataset shape: " + str(next(generator)[0][0].shape))
        print("Evaluating on " + str(nb_validation_samples) + " test batches ...")

        target_names = prop_utils.load_json(class_path).values()
        ground_truth = []
        for _ in range(0, nb_validation_samples):
            b = next(generator)
            ground_truth.extend(b[1].numpy())
        ground_truth = np.array(ground_truth).reshape(len(ground_truth))
        Y_pred = self._model.predict_generator(generator, nb_validation_samples)
        y_pred = np.argmax(Y_pred, axis=1)
        con_mat = tf.math.confusion_matrix(labels=ground_truth, predictions=y_pred).numpy()
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        con_mat_df = pd.DataFrame(con_mat_norm, index=target_names, columns=target_names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, linewidths=1)
        #plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        # fix for mpl bug that cuts off top/bottom of seaborn viz
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
        plt.savefig(os.path.join(log_path, "cm.png"), format='png')
        report = classification_report(ground_truth, y_pred, target_names=target_names)

        val_loss, val_acc = self._model.evaluate_generator(generator, nb_validation_samples/batch_size)

        if False:
            fpr, tpr, threshold = metrics.roc_curve(ground_truth, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig(os.path.join(log_path, "roc.png"), format='png')

        f = open(os.path.join(log_path, "log.txt"), "w+")
        f.write("Classification Report: \n" + str(report) + "\n")
        f.write("Evaluation Loss: " + str(val_loss) + "\n")
        f.write("Evaluation Acc.: " + str(val_acc))
        f.close()
        return val_loss, val_acc

    def predict(self, img):
        """ Predicts the content of an image.
        :param img:
        :return: predicted label
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Unable to predict type {}".format(type(img)))
        prediction = self._model.predict(np.array([img, ]))
        return prediction[0][0]

    def print(self):
        """ Prints the current learn setup in the console
        """
        self._model.summary()
