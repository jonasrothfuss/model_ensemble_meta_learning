import numpy as np


class DataBuffer(object):

    def __init__(self, example, max_buffer=int(1e6)):
        assert type(example) is dict
        self.keys = example.keys()
        self._buffer = dict([(k, np.zeros((max_buffer,)+v.shape, dtype=v.dtype)) for k, v in example.items()])
        self.max_buffer = max_buffer

    def add_data(self, data):
        for k, v in data.items():
            self._buffer[k] = np.concatenate([self._buffer[k], v])[-self.max_buffer:]

    def get_data(self):
        return self._buffer  # TODO: Should I copy here?

