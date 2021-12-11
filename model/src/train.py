import keras
from numpy.lib import utils
import model
import sys
import os
import numpy as np
import typing
import random
import tensorflow as tf
import datetime

# import infrastructure
ROOT_SCRIPTS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'scripts'))
sys.path.append(ROOT_SCRIPTS_PATH)
import media_descriptor  # nopep8
import media_audio  # nopep8
import metadata_extractor  # nopep8
import utils # nopep8


def load_data(path: str) -> typing.Tuple[media_audio.MediaAudio, media_descriptor.MediaDescriptor]:
    '''
    Loads audio file and associated metadata
    '''
    data = media_audio.MediaAudio()
    # print(path)
    data.load(path)

    metadata = media_descriptor.MediaDescriptor(path.replace('.wav', '.json'))
    metadata.read()
    return data, metadata


def prepare_data(data: media_audio.MediaAudio, metadata: media_descriptor.MediaDescriptor, 
                input_size: int, window_overlap: float) -> typing.Tuple[np.array, np.array]:
    '''
    Prepare data so in can be fed into training phase
    '''
    output_size = len(metadata_extractor.ALLOWED_CLASSES.keys())
    audio = data.y
    label = metadata_extractor.translate_label(
        metadata_extractor.ALLOWED_CLASSES, metadata.data()['label'])

    X = []
    Y = []

    while len(audio) < input_size:
        audio = np.concatenate((audio, audio), axis=None)

    for i in range(input_size, len(audio), int(input_size * window_overlap)):
        y = np.zeros((output_size,))
        y[label] = 1
        Y.append(y)

        x = np.expand_dims(np.array(audio[(i - input_size):i]), axis=-1)
        X.append(x)

    return np.array(X), np.array(Y)


def prepare_batch(trainlist, batch_size, batch_index, dnn_input_size, window_overlap):
    idx_start = batch_index * batch_size
    idx_end = idx_start + batch_size
    batch_list = trainlist[idx_start:idx_end]

    X = None
    Y = None
    if len(batch_list) == 0:
        return None

    for file in batch_list:
        data, metadata = load_data(file)
        x, y = prepare_data(data, metadata, dnn_input_size, window_overlap)

        if X is None:
            X = x
            Y = y
        else:
            X = np.vstack((X, x))
            Y = np.vstack((Y, y))
    return X, Y

def prepare_experiment(experiment_tag):
    experiment_folder = os.path.join(utils.make_path_absolute('experiments'), experiment_tag)
    os.makedirs(experiment_folder)

    checkpoint_folder = os.path.join(experiment_folder, 'checkpoints')
    os.makedirs(checkpoint_folder)

    logs_folder = os.path.join(experiment_folder, 'logs')
    os.makedirs(logs_folder)

    return experiment_folder, checkpoint_folder, logs_folder

def train(trainlist):
    keras.backend.clear_session()
    params = utils.load_params()
    factory = model.get_factory(params)
    m = factory.build_model()
    print(m.summary())

    epochs = params['epochs']
    batch_size = params['batch_size']
    file_batch_size = params['file_batch_size']
    keep_checkpoint_at_every_n_fit = params['keep_checkpoint_at_every_n_fit']
    window_overlap = params['window_overlap']

    fit_count = 0
    for epoch_index in range(epochs):
        random.shuffle(trainlist)

        file_batch_total = int(len(trainlist)/file_batch_size)
        for file_batch_index in range(0, file_batch_total):
            ret = prepare_batch(trainlist, file_batch_size, file_batch_index, factory.INPUT_SIZE, window_overlap)

            if ret is None:
                break

            X, Y = ret

            idxs = tf.random.shuffle(tf.range(X.shape[0]))
            X = tf.gather(X, idxs)
            Y = tf.gather(Y, idxs)
            # print(X.shape)
            # print(Y.shape)
            # break
            print('epoch: %d - %d/%d' %
                  (epoch_index + 1, file_batch_index + 1, file_batch_total))
            history = m.fit(X, Y, batch_size=batch_size)
            fit_count += 1

            if (keep_checkpoint_at_every_n_fit > 0) and (fit_count % keep_checkpoint_at_every_n_fit == 0):
                # save model here
                pass

        # checkpoint_name = '%05d.h5' % (epoch_index + 1)
        # checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)
        # print('Saving checkpoint "%s"' % checkpoint_path)
        # m.save(checkpoint_path)


trainlist = []
with open('datasets/trainlist.txt', 'r') as f:
    for line in f.readlines():
        trainlist.append(line.replace('\n', ''))

train(trainlist)
