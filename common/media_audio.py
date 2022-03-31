import librosa
import os
import hashlib
from common.utils import *

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


def list_media_files(dir, recursively=False):
    list_files = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if '.wav' not in f:
                continue

            fullpath = os.path.join(root, f)
            if not os.path.exists(fullpath.replace('.wav', '.json')):
                continue

            list_files.append(make_path_relative(fullpath))
        
        if recursively == False:
            break
    return list_files

class MediaAudio(BaseAudio):
    def __init__(self) -> None:
        BaseAudio.__init__(self, None, None, None)
        self.filepath = ''

    def load(self, path):
        self.filepath = path.replace(PROJECT_PATH, '')
        self.y, self.sr = librosa.load(path, sr=None)
        self.duration = librosa.get_duration(filename=path)
