import numpy as np
import os
import sys
import typing
import random
import tensorflow as tf

# import infrastructure
from common import media_descriptor  # nopep8
from common import media_audio  # nopep8
from common import metadata_extractor  # nopep8


def load_data(path: str) -> typing.Tuple[media_audio.MediaAudio, media_descriptor.MediaDescriptor]:
    '''
    Loads audio file and associated metadata
    '''
    data = media_audio.MediaAudio()
    data.load(path)

    metadata = media_descriptor.MediaDescriptor(path.replace('.wav', '.json'))
    metadata.read()
    return data, metadata


def prepare_data(data: media_audio.MediaAudio, metadata: media_descriptor.MediaDescriptor,
                 input_size: int, window_overlap: float) -> typing.Tuple[np.array, np.array]:
    '''
    Prepare data so in can be fed into training phase
    '''
    output_size = len(metadata_extractor.ALLOWED_CLASSES.keys()) - 1
    audio = data.y
    label = metadata_extractor.translate_label(
        metadata_extractor.ALLOWED_CLASSES, metadata.data()['label'])

    X = []
    Y = []

    while len(audio) < input_size:
        audio = np.concatenate((audio, audio), axis=None)

    for i in range(input_size, len(audio) + 1, int(input_size * (1.0 - window_overlap))):
        y = np.zeros((output_size,))
        if label < output_size:
            y[label] = 1
        Y.append(y)

        x = np.expand_dims(np.array(audio[(i - input_size):i]), axis=-1)
        X.append(x)

    assert len(x) > 0
    assert len(y) > 0

    return np.array(X), np.array(Y)


def prepare_batch(filelist, batch_size, batch_index, dnn_input_size, window_overlap):
    idx_start = batch_index * batch_size
    idx_end = idx_start + batch_size
    batch_list = filelist[idx_start:idx_end]

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


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list: list, file_batch_size: int, input_size: int, window_overlap: float) -> None:
        super().__init__()
        self.file_list = file_list
        self.file_batch_size = file_batch_size
        self.input_size = input_size

        assert window_overlap > 0 and window_overlap < 1
        self.window_overlap = window_overlap

    def get_batch_item(self, item, shuffle=True):
        ret = prepare_batch(self.file_list, self.file_batch_size,
                            item, self.input_size, self.window_overlap)

        assert ret is not None
        X, Y = ret

        if shuffle:
            idxs = tf.random.shuffle(tf.range(X.shape[0]))
            X = tf.gather(X, idxs)
            Y = tf.gather(Y, idxs)
        return X, Y

    def __getitem__(self, item):
        return self.get_batch_item(item)

    def __len__(self):
        return int(np.floor(len(self.file_list)/self.file_batch_size))

    def on_epoch_end(self):
        random.shuffle(self.file_list)


class RandomWindowAudioGenerator:
    def __init__(self, file_list: list, window_size: int, max_resample_file=5, **kwargs):
        self.file_list = file_list
        self.window_size = window_size
        self.max_resample_file = max_resample_file
        self.data_aug = kwargs.get('data_aug', None)

    def __adjust_audio_length(self, audio) -> np.array:
        len_audio = len(audio)
        if self.window_size <= len_audio:
            # no need to adjust size
            return audio

        # make a fixed increment, so window size is always smaller than audio length
        size_diff = self.window_size - len_audio + 10
        prepend_size = np.random.randint(0, size_diff)
        append_size = size_diff - prepend_size

        audio = np.concatenate((audio, np.zeros((append_size,))))
        audio = np.concatenate((np.zeros((prepend_size,)), audio))
        return audio

    def __select_window(self, available_size) -> tuple:
        max_start_idx = available_size - self.window_size - 1
        start_idx = np.random.randint(0, max_start_idx)
        end_idx = start_idx + self.window_size
        return (start_idx, end_idx)

    def next(self):
        resample_file = max(1, self.max_resample_file)
        output_size = len(metadata_extractor.ALLOWED_CLASSES.keys()) - 1
        for file in self.file_list:
            for n in range(resample_file):
                data, metadata = load_data(file)
                audio = data.y

                if self.data_aug:
                    audio = self.data_aug.augment(audio)

                label = metadata_extractor.translate_label(
                    metadata_extractor.ALLOWED_CLASSES, metadata.data()['label'])

                audio = self.__adjust_audio_length(audio)
                start, end = self.__select_window(len(audio))

                label_onehot = np.zeros((output_size,))
                if label < output_size:
                    label_onehot[label] = 1
                yield (np.expand_dims(audio[start:end], axis=-1), label_onehot)


class DatasetLoader:
    def __init__(self, file_list: list, batch_size: int, input_size: int, window_overlap: float, **kwargs) -> None:
        self.file_list = file_list
        self.batch_size = batch_size
        self.input_size = input_size
        self.window_overlap = window_overlap
        self.data_aug = kwargs.get('data_aug', None)

    def dataset(self) -> tf.data.Dataset:
        max_resample_file = 5
        gen = RandomWindowAudioGenerator(
            self.file_list, self.input_size, max_resample_file=max_resample_file, data_aug=self.data_aug)

        ds = tf.data.Dataset.from_generator(gen.next,
                                            output_types=(
                                                tf.float32, tf.int32),
                                            output_shapes=(tf.TensorShape([self.input_size, 1]), tf.TensorShape([2, ])))
        ds = ds.shuffle(buffer_size=self.batch_size*(max_resample_file*10))
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def on_epoch_end(self):
        random.shuffle(self.file_list)
