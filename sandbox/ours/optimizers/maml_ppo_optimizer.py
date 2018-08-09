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
    
    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            step_size=1e-3,
            multi_adam=1,
            **kwargs):
        Serializable.quick_init(self, locals())
        super().__init__(
                            tf_optimizer_cls=tf_optimizer_cls,
                            tf_optimizer_args=tf_optimizer_args,
                            step_size=step_size,
                            **kwargs
                        )
        self.multi_adam = multi_adam
        if self.multi_adam > 1:
            if tf_optimizer_cls is None:
                tf_optimizer_cls = tf.train.AdamOptimizer
            if tf_optimizer_args is None:
                tf_optimizer_args = dict(learning_rate=step_size)
            self._tf_optimizers = [tf_optimizer_cls(**tf_optimizer_args) for _ in range(multi_adam)]
            # No init tf optimizer right now
        
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
        if self.multi_adam > 1:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                # for batch norm
                updates = tf.group(*update_ops)
                with tf.control_dependencies([updates]):
                    self._train_ops = [optimizer.minimize(loss, var_list=target.get_params(trainable=True)) for optimizer in self._tf_optimizers]
            else:
                self._train_ops = [optimizer.minimize(loss, var_list=target.get_params(trainable=True)) for optimizer in self._tf_optimizers]
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

            for j, batch in enumerate(dataset.iterate(update=True)):
                if self._init_train_op is not None:
                    sess.run(self._init_train_op, dict(list(zip(self._input_vars, batch))))
                    self._init_train_op = None  # only use it once
                else:
                    if self.multi_adam > 1:
                        sess.run(self._train_ops[epoch * self._batch_size + j], dict(list(zip(self._input_vars, batch))))
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
