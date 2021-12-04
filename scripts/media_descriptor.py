import json
import os
from copy import deepcopy

class MediaDescriptor:
    '''
    Metadata associated with a single media file
    '''
    def __init__(self, path) -> None:
        self.path = path
        self.__data = None

    def update(self, new_data):
        if os.path.exists(self.path):
            self.__data = self.read()
            self.__data.update(new_data)
        else:
            self.__data = new_data
        with open(self.path, 'w') as f:
            json.dump(self.__data, f)

    def read(self):
        with open(self.path, 'r') as f:
            self.__data = json.load(f)
        return self.data()

    def data(self):
        return deepcopy(self.__data)
