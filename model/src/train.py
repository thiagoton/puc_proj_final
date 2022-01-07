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


def prepare_experiment(experiment_tag):
    experiment_folder = os.path.join(
        utils.make_path_absolute('experiments'), experiment_tag)
    os.makedirs(experiment_folder)

    checkpoint_folder = os.path.join(experiment_folder, 'checkpoints')
    os.makedirs(checkpoint_folder)

    logs_folder = os.path.join(experiment_folder, 'logs')
    os.makedirs(logs_folder)

    return experiment_folder, checkpoint_folder, logs_folder


def make_dvc_checkpoint(m):
    root = utils.make_path_absolute('experiments')
    last_path = os.path.join(root, 'last')
    os.makedirs(last_path, exist_ok=True)
    ckpt_path = os.path.join(last_path, 'checkpoint.h5')
    m.save(ckpt_path)
    dvc.api.make_checkpoint()

def save_model(m, ckpt_folder, epoch, file_batch_index=0, fit_count=0):
    checkpoint_name = 'ckpt_%05d-%05d-%05d.h5' % (
        epoch, file_batch_index, fit_count)
    checkpoint_path = os.path.join(ckpt_folder, checkpoint_name)
    print('Saving checkpoint "%s"' % checkpoint_path)
    m.save(checkpoint_path)

    make_dvc_checkpoint(m)


def train(trainlist, validationlist=[]):
    keras.backend.clear_session()
    params = utils.load_params()
    factory = model.get_factory(params)
    m = factory.build_model()
    print(m.summary())

    epochs = params['epochs']
    batch_size = params['batch_size']
    file_batch_size = params['file_batch_size']
    keep_checkpoint_at_every_n_epoch = params['keep_checkpoint_at_every_n_epoch']
    window_overlap = params['window_overlap']
    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 2
        print('Number of workers set to 2')

    experiment_tag = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_folder, checkpoint_folder, logs_folder = prepare_experiment(
        experiment_tag)

    if len(validationlist):
        tensorboard_cb = callbacks.TensorBoard(log_dir=logs_folder, histogram_freq=1)
    else:
        tensorboard_cb = callbacks.TensorBoard(log_dir=logs_folder)

    data_gen = DataGenerator(trainlist, file_batch_size,
                             factory.INPUT_SIZE, window_overlap)
    val_data_gen = None
    if len(validationlist):
        val_data_gen = DataGenerator(validationlist,
                                     file_batch_size,
                                     factory.INPUT_SIZE, window_overlap)
    for epoch_index in range(epochs):
        print('epoch: %d' % (epoch_index + 1))
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

        with open('metrics.json', 'a') as fd:
            json.dump(history.history, fd)
            fd.write('\n')

        if (keep_checkpoint_at_every_n_epoch > 0) and (epoch_index % keep_checkpoint_at_every_n_epoch == 0):
            save_model(m, checkpoint_folder, epoch_index + 1)


trainlist = []
with open('datasets/trainlist.txt', 'r') as f:
    for line in f.readlines():
        trainlist.append(line.replace('\n', ''))

validationlist = []
with open('datasets/testlist.txt', 'r') as f:
    for line in f.readlines():
        validationlist.append(line.replace('\n', ''))

train(trainlist, validationlist)
