import librosa
import os
import hashlib

PROJECT_PATH = os.path.realpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..'))


def make_path_relative(input_path):
    return os.path.relpath(os.path.realpath(input_path), PROJECT_PATH)


class BaseAudio:
    '''
    Base audio abstraction (signal, sampling rate and duration).
    This class should not be used directly
    '''

    def __init__(self, y, sr, duration) -> None:
        self.y = y
        self.sr = sr
        self.duration = duration

    def get_signature(self):
        ''''
        Returns a hash for the given audio
        '''
        return hashlib.md5(self.y).hexdigest()

    def get_descriptor(self):
        '''
        Returns a dict describing the current object
        '''
        descriptor = self.__dict__.copy()
        descriptor.pop('y')
        descriptor['signature'] = self.get_signature()
        return descriptor


class PreprocessedAudio(BaseAudio):
    '''
    Preprocessed audio type. Used after audio is preprocessed
    '''

    def __init__(self, y, sr, duration, raw_filepath) -> None:
        BaseAudio.__init__(self, y, sr, duration)
        self.raw_filepath = raw_filepath
        self.label = None

    def get_signature(self):
        ''''
        Returns a hash for the given audio
        '''
        return hashlib.md5(self.y).hexdigest()


class MediaAudio(BaseAudio):
    def __init__(self) -> None:
        BaseAudio.__init__(self, None, None, None)
        self.filepath = ''

    def load(self, path):
        self.filepath = path.replace(PROJECT_PATH, '')
        self.y, self.sr = librosa.load(path, sr=None)
        self.duration = librosa.get_duration(filename=path)
