import keras
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


def prepare_experiment(experiment_tag) -> tuple():
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
        utils.make_path_absolute('model'), 'train')
    os.makedirs(checkpoint_folder, exist_ok=True)

    return experiment_folder, checkpoint_folder, logs_folder


def save_model(m: keras.Model, save_folder: str, model_name: str):
    model_path = os.path.join(save_folder, model_name)
    m.save(model_path)
    return model_path


def save_checkpoint(m: keras.Model, epoch: int, checkpoint_folder: str, state={}):
    checkpoint_name = 'checkpoint.h5'
    save_model(m, checkpoint_folder, checkpoint_name)
    with open(os.path.join(checkpoint_folder, 'checkpoint.json'), 'w') as fd:
        json.dump(state, fd)
    dvc.api.make_checkpoint()


def save_best(m: keras.Model, save_folder: str, state={}, model_name='best'):
    save_model(m, save_folder, model_name + '.h5')
    with open(os.path.join(save_folder, model_name + '.json'), 'w') as fd:
        json.dump(state, fd)
    dvc.api.make_checkpoint()


def train(trainlist, validationlist=[]):
    keras.backend.clear_session()
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
        experiment_tag)

    if len(validationlist):
        tensorboard_cb = callbacks.TensorBoard(
            log_dir=logs_folder, histogram_freq=1)
    else:
        tensorboard_cb = callbacks.TensorBoard(log_dir=logs_folder)

    data_gen = DataGenerator(trainlist, file_batch_size,
                             factory.INPUT_SIZE, window_overlap)
    val_data_gen = None
    if len(validationlist):
        val_data_gen = DataGenerator(validationlist,
                                     file_batch_size,
                                     factory.INPUT_SIZE, window_overlap)
    best_acc = 0
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

        history = m.fit(data_gen,
                        batch_size=batch_size,
                        callbacks=[tensorboard_cb],
                        initial_epoch=epoch_index,
                        epochs=epoch_index+1,
                        validation_data=val_data,
                        verbose=1)
        acc = history.history['accuracy'][0]
        loss = history.history['loss'][0]
        val_acc = history.history['val_accuracy'][0]
        val_loss = history.history['val_loss'][0]
        if val_acc > best_acc:
            best_acc = val_acc
            save_best(m, checkpoint_folder, CheckpointState(
                epoch_index, val_acc, val_loss).asJson(), model_name='best_' + model_name)

        if (keep_checkpoint_at_every_n_epoch > 0) and (epoch_index % keep_checkpoint_at_every_n_epoch == 0):
            save_checkpoint(m, epoch_index, checkpoint_folder,
                            CheckpointState(epoch_index, acc, loss).asJson())


trainlist = utils.load_filelist('datasets/trainlist.txt')
validationlist = utils.load_filelist('datasets/testlist.txt')

train(trainlist, validationlist)
