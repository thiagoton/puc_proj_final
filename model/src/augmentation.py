
import random
import numpy as np
import logging

logger = logging.getLogger()

AVAILABLE_OPERATIONS = {}


def register_operation(op_name, op_class):
    AVAILABLE_OPERATIONS[op_name] = op_class


class GenericOperation:
    def __init__(self, context: dict) -> None:
        self.prob = context['prob']

    def apply(self, data) -> np.array:
        if random.random() < self.prob:
            if type(data) is list:
                data = np.array(data)
            return self.do_apply(data)
        else:
            return data

    def do_apply(self, data: np.array) -> np.array:
        raise NotImplementedError()


class MultiplierOperation(GenericOperation):
    def __init__(self, context: dict) -> None:
        super().__init__(context)
        self.max = context['max']
        self.min = context['min']

    def do_apply(self, data: np.array) -> np.array:
        factor = random.uniform(self.min, self.max)
        data = factor * data
        data[data < -1] = -1
        data[data > 1] = 1
        return data


register_operation('multiplier', MultiplierOperation)


class WhiteNoiseOperation(GenericOperation):
    def __init__(self, context: dict) -> None:
        super().__init__(context)
        self.sigma_max = context['sigma_max']
        self.sigma_min = context['sigma_min']

    def do_apply(self, data: np.array) -> np.array:
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        data += (sigma**2) * np.random.randn(*(data.shape))
        data[data < -1] = -1
        data[data > 1] = 1
        return data


register_operation('white_noise', WhiteNoiseOperation)


class DataAugmenter:
    def __init__(self, context: dict) -> None:
        self.enabled = context['enabled']
        self.operations = []

        if self.enabled:
            for op_name in context['order']:
                params = context['operations'][op_name]
                assert op_name in AVAILABLE_OPERATIONS.keys(), 'operation %s not found. Allowed operations=%s' % (
                    op_name, str(AVAILABLE_OPERATIONS.keys()))
                self.operations.append(AVAILABLE_OPERATIONS[op_name](params))

    def augment(self, data: np.array) -> np.array:
        for op in self.operations:
            data = op.apply(data)
        return data
