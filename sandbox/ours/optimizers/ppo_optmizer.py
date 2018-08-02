from rllab_maml.misc import ext
from rllab_maml.misc import logger
from rllab_maml.core.serializable import Serializable
from sandbox_maml.rocky.tf.misc import tensor_utils
from sandbox_maml.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
# from rllab_maml.algo.first_order_method import parse_update_method
from rllab_maml.optimizers.minibatch_dataset import BatchDataset, MAMLBatchDataset
from collections import OrderedDict
import tensorflow as tf
import time
from functools import partial
import pyprind

class MAMLPPOOptimizer(FirstOrderOptimizer):
    ## Right now it's just implemented one gradient step with all the data
    
    def update_opt(self, loss, target, inputs, inner_kl, outer_kl, extra_inputs=None, meta_batch_size=1, num_grad_updates=1, **kwargs):
        """
        :param inner_kl: Symbolic expression for inner kl
        :param outer_kl: Symbolic expression for outer kl
        :param meta_batch_size: number of MAML tasks, for batcher
        """
        super().update_opt(loss, target, inputs, extra_inputs, **kwargs)
        if extra_inputs is None:
            extra_inputs = list()
        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, loss),
            f_inner_kl=lambda: tensor_utils.compile_function(inputs + extra_inputs, inner_kl),
            f_outer_kl=lambda: tensor_utils.compile_function(inputs + extra_inputs, outer_kl),
        )
        self.meta_batch_size = meta_batch_size
        self.num_grad_updates = num_grad_updates

    def inner_kl(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_inner_kl"](*(tuple(inputs) + extra_inputs))

    def outer_kl(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_outer_kl"](*(tuple(inputs) + extra_inputs))


    def optimize(self, inputs, extra_inputs=None, callback=None):

        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        f_loss = self._opt_fun["f_loss"]

        if extra_inputs is None:
            extra_inputs = tuple()

        last_loss = f_loss(*(tuple(inputs) + extra_inputs))

        start_time = time.time()
        # Overload self._batch size
        dataset = MAMLBatchDataset(inputs, num_batches=self._batch_size, extra_inputs=extra_inputs, meta_batch_size=self.meta_batch_size, num_grad_updates=self.num_grad_updates)

        sess = tf.get_default_session()
        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))
                progbar = pyprind.ProgBar(len(inputs[0]))

            for batch in dataset.iterate(update=True):
                if self._init_train_op is not None:
                    sess.run(self._init_train_op, dict(list(zip(self._input_vars, batch))))
                    self._init_train_op = None  # only use it once
                else:
                    sess.run(self._train_op, dict(list(zip(self._input_vars, batch))))

                if self._verbose:
                    progbar.update(len(batch[0]))

            if self._verbose:
                if progbar.active:
                    progbar.stop()

            new_loss = f_loss(*(tuple(inputs) + extra_inputs))

            if self._verbose:
                logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))
            if self._callback or callback:
                elapsed = time.time() - start_time
                callback_args = dict(
                    loss=new_loss,
                    params=self._target.get_param_values(trainable=True) if self._target else None,
                    itr=epoch,
                    elapsed=elapsed,
                )
                if self._callback:
                    self._callback(callback_args)
                if callback:
                    callback(**callback_args)

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss
