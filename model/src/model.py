from typing import Sequence
import keras
from keras.layers import *

AVAILABLE_FACTORIES = {}
def register_factory(class_type: type):
    AVAILABLE_FACTORIES[class_type.__name__] = class_type
class BaseFactory:
    def __init__(self) -> None:
        pass

    def build_model(self, **kwargs):
        raise NotImplementedError()

class TimeDistributedCnnLstm(BaseFactory):
    def __init__(self, **params) -> None:
        super().__init__()
        self.INPUT_SIZE = int(params.get('input_size', 128))
        self.TIME_WINDOW_SIZE = int(params.get('time_window_size', 5))

    def build_model(self, **kwargs):
        model = keras.Sequential()

        op = Conv1D(32, 16, strides=1, padding='causal')
        model.add(TimeDistributed(op, input_shape=(
            self.TIME_WINDOW_SIZE, self.INPUT_SIZE, 1)))

        op = MaxPooling1D(pool_size=2)
        model.add(TimeDistributed(op))

        op = Conv1D(32, 16, strides=1, padding='causal')
        model.add(TimeDistributed(op))

        op = Flatten()
        model.add(TimeDistributed(op))

        model.add(LSTM(5))
        model.add(Dense(3))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model
register_factory(TimeDistributedCnnLstm)

class Cnn(BaseFactory):
    def __init__(self, **params) -> None:
        super().__init__()
        self.INPUT_SIZE = int(params.get('input_size', 128))

    def build_model(self, **kwargs):
        model = keras.Sequential()

        op = Conv1D(32, 16, padding='causal', input_shape=(self.INPUT_SIZE, 1))
        model.add(op)

        op = Dense(64)
        model.add(op)

        op = MaxPooling1D(pool_size=2)
        model.add(op)

        op = BatchNormalization()
        model.add(op)
        
        op = Conv1D(32, 16, padding='causal')
        model.add(op)

        op = Dense(64)
        model.add(op)

        op = MaxPooling1D(pool_size=2)
        model.add(op)

        op = BatchNormalization()
        model.add(op)

        op = Flatten()
        model.add(op)

        model.add(Dense(3))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model
register_factory(Cnn)

class WaveNet(BaseFactory):
    def __init__(self, **params) -> None:
        super().__init__()
        input_window = float(params.get('input_window', 1.5))
        self.INPUT_SIZE = int(input_window * 24000)

    def build_model(self, **kwargs):
        model = keras.Sequential()

        he_normal = lambda: keras.initializers.HeNormal()

        op = Conv1D(filters=64, kernel_size=64, padding='same', input_shape=(self.INPUT_SIZE, 1), kernel_initializer=he_normal())
        model.add(op)

        op = Conv1D(filters=64, kernel_size=64, padding='same', kernel_initializer=he_normal())
        model.add(op)

        op = MaxPooling1D(pool_size=220)
        model.add(op)

        op = Dropout(rate=0.1)
        model.add(op)

        op = Reshape((64, 163, 1))
        model.add(op)

        op = Conv2D(filters=32, kernel_size=(4, 4), padding='same', strides=(1,1), kernel_initializer=he_normal())
        model.add(op)

        op = Conv2D(filters=32, kernel_size=(4, 4), padding='same', strides=(1,1), kernel_initializer=he_normal())
        model.add(op)

        op = MaxPooling2D(pool_size=(2,2), strides=(2,2))
        model.add(op)

        op = Dropout(rate=0.1)
        model.add(op)

        op = Conv2D(filters=64, kernel_size=(4, 4), padding='same', strides=(1,1), kernel_initializer=he_normal())
        model.add(op)

        op = Conv2D(filters=64, kernel_size=(4, 4), padding='same', strides=(1,1), kernel_initializer=he_normal())
        model.add(op)

        op = MaxPooling2D(pool_size=(2,2), strides=(2,2))
        model.add(op)

        op = Dropout(rate=0.1)
        model.add(op)

        op = Flatten()
        model.add(op)

        model.add(Dense(512))
        model.add(Dense(64))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model
register_factory(WaveNet)

def build_model(params):
    model_name = params['model_name']
    assert model_name in AVAILABLE_FACTORIES.keys()
    factory_type = AVAILABLE_FACTORIES[model_name]
    factory = factory_type(**params[model_name])
    return factory.build_model()