import numpy as np


class BatchDataset(object):

    def __init__(self, inputs, batch_size, extra_inputs=None):
        self._inputs = [
            i for i in inputs
        ]
        if extra_inputs is None:
            extra_inputs = []
        self._extra_inputs = extra_inputs
        self._batch_size = batch_size
        if batch_size is not None:
            self._ids = np.arange(self._inputs[0].shape[0])
            self.update()

    @property
    def number_batches(self):
        if self._batch_size is None:
            return 1
        return int(np.ceil(self._inputs[0].shape[0] * 1.0 / self._batch_size))

    def iterate(self, update=True):
        if self._batch_size is None:
            yield list(self._inputs) + list(self._extra_inputs)
        else:
            for itr in range(self.number_batches):
                batch_start = itr * self._batch_size
                batch_end = (itr + 1) * self._batch_size
                batch_ids = self._ids[batch_start:batch_end]
                batch = [d[batch_ids] for d in self._inputs]
                yield list(batch) + list(self._extra_inputs)
            if update:
                self.update()

    def update(self):
        np.random.shuffle(self._ids)

class MAMLBatchDataset(object):
    def __init__(self, inputs, num_batches=1, extra_inputs=None, meta_batch_size=1, num_grad_updates=1):
        self._inputs = [
            i for i in inputs
        ]
        if extra_inputs is None:
            extra_inputs = []
        self._extra_inputs = extra_inputs
        num_sets = num_grad_updates + 1 # How many sets of trajectories collected
        set_size = len(inputs) // num_sets # Number of entries (obs, actions, rewards, means, stds) per set
        self.number_batches = num_batches
        if num_batches > 1:
            self._ids = [[np.arange(self._inputs[i + j * set_size].shape[0]) for i in range(meta_batch_size)] for j in range(num_sets)]
            self.update()
            self._batch_size = min([min([i.shape[0] for i in j]) for j in self._ids]) // num_batches

    def iterate(self, update=True):
        if self.number_batches == 1:
            yield list(self._inputs) + list(self._extra_inputs)
        else:
            set_size = len(self._inputs) // len(self._ids)
            meta_batch_size = len(self._ids[0])
            inputs_per_task = set_size // meta_batch_size - 1 # Account for distr_var coming in pairs
            for itr in range(self.number_batches):
                batch_start = itr * self._batch_size
                batch_end = (itr + 1) * self._batch_size
                batch = [None] * len(self._inputs)
                for i, j in enumerate(self._ids):
                    for k, ids in enumerate(j):
                        batch_ids = ids[batch_start:batch_end]
                        for input_type in range(inputs_per_task):
                            if input_type < inputs_per_task - 1:
                                input_ind = i * set_size + input_type * meta_batch_size + k
                                batch[input_ind] = self._inputs[input_ind][batch_ids]
                            else:
                                input_ind = i * set_size + input_type * meta_batch_size + 2 * k
                                batch[input_ind] = self._inputs[input_ind][batch_ids]
                                batch[input_ind + 1] = self._inputs[input_ind + 1][batch_ids]
                yield list(batch) + list(self._extra_inputs)
            if update:
                self.update()

    def update(self):
        for j in self._ids:
            for ids in j:
                np.random.shuffle(ids)