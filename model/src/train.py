import tensorflow as tf
from numpy.lib import utils
from tensorflow.keras import callbacks
import model
import sys
import os
import numpy as np
# Import TensorBoard
import datetime
from generator import *
import json
import dvc.api
import time
import logger

# import infrastructure
ROOT_SCRIPTS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'scripts'))
sys.path.append(ROOT_SCRIPTS_PATH)
import media_descriptor  # nopep8
import media_audio  # nopep8
import metadata_extractor  # nopep8
import utils  # nopep8


class CheckpointState:
    def __init__(self, epoch, acc, loss):
        self.epoch = epoch
        self.acc = acc
        self.loss = loss

    def asJson(self):
        return self.__dict__()

    def __dict__(self):
        return {'epoch': self.epoch, 'accuracy': self.acc, 'loss': self.loss}


def prepare_experiment(experiment_tag, model_name) -> tuple():
    '''
    Setup the folder structure needed for the training
    return: experiment_folder, checkpoint_folder, logs_folder
    '''
    experiment_folder = os.path.join(
        utils.make_path_absolute('experiments'), experiment_tag)
    os.makedirs(experiment_folder)

    logs_folder = os.path.join(experiment_folder, 'logs')
    os.makedirs(logs_folder)

    checkpoint_folder = os.path.join(
        utils.make_path_absolute('model'), 'train', model_name)
    os.makedirs(checkpoint_folder, exist_ok=True)

    return experiment_folder, checkpoint_folder, logs_folder


def train(trainlist, validationlist=[]):
    tf.keras.backend.clear_session()
    params = utils.load_params()
    factory = model.get_factory(params)
    m = factory.build_model()
    print(m.summary())

    model_name = params['model_name']
    epochs = params['epochs']
    batch_size = params['batch_size']
    file_batch_size = params['file_batch_size']
    keep_checkpoint_at_every_n_epoch = params['keep_checkpoint_at_every_n_epoch']
    window_overlap = params['window_overlap']
    experiment_tag = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_folder, checkpoint_folder, logs_folder = prepare_experiment(
        experiment_tag, model_name)

    log = logger.Logger(checkpoint_folder)

    if len(validationlist):
        tensorboard_cb = callbacks.TensorBoard(
            log_dir=logs_folder, histogram_freq=1)
    else:
        tensorboard_cb = callbacks.TensorBoard(log_dir=logs_folder)

    data_loader = DatasetLoader(trainlist, batch_size, factory.INPUT_SIZE, window_overlap)
    val_data_gen = None
    if len(validationlist):
        val_data_gen = DataGenerator(validationlist,
                                     file_batch_size,
                                     factory.INPUT_SIZE, window_overlap)
    best_acc = 0
    cummulative_time = 0
    for epoch_index in range(epochs):
        print('epoch: %d' % (epoch_index))
        val_data = None
        if val_data_gen:
            val_x = None
            val_y = None
            for n in range(5):
                x, y = val_data_gen[n]
                if val_x is None:
                    val_x = x
                    val_y = y
                else:
                    val_x = np.vstack((val_x, x))
                    val_y = np.vstack((val_y, y))
            val_data = (val_x, val_y)

            val_data_gen.on_epoch_end()

        t0 = time.time()
        history = m.fit(data_loader.dataset(),
                        callbacks=[tensorboard_cb],
                        initial_epoch=epoch_index,
                        epochs=epoch_index+1,
                        validation_data=val_data,
                        verbose=1)
        data_loader.on_epoch_end()

        do_checkpoint = False
        t1 = time.time()
        epoch_time = t1 - t0
        cummulative_time += epoch_time
        epoch_avg_time = cummulative_time/(epoch_index + 1)
        print("Epoch took %.3fs (avg=%.3fs)" % (epoch_time, epoch_avg_time))
        acc = history.history['accuracy'][0]
        loss = history.history['loss'][0]
        val_acc = history.history['val_accuracy'][0]
        val_loss = history.history['val_loss'][0]

        log.log_timeseries('epoch_time', epoch_time, epoch_index)
        log.log_timeseries('epoch_avg_time', epoch_avg_time, epoch_index)
        log.log_timeseries('accuracy', acc, epoch_index)
        log.log_timeseries('loss', loss, epoch_index)
        log.log_timeseries('val_acc', val_acc, epoch_index)
        log.log_timeseries('val_loss', val_loss, epoch_index)

        if val_acc > best_acc:
            best_acc = val_acc
            log.save_model(m, 'best.h5')
            log.log_metric(CheckpointState(epoch_index, val_acc, val_loss).asJson(), filename='best.json')
            do_checkpoint = True

        if (keep_checkpoint_at_every_n_epoch > 0) and (epoch_index % keep_checkpoint_at_every_n_epoch == 0):
            log.save_model(m, 'checkpoint.h5')
            log.log_metric(CheckpointState(epoch_index, acc, loss).asJson(), filename='checkpoint.json')
            do_checkpoint = True

        if do_checkpoint:
            dvc.api.make_checkpoint()


trainlist = utils.load_filelist('datasets/trainlist.txt')
validationlist = utils.load_filelist('datasets/testlist.txt')

train(trainlist, validationlist)
