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

class PPOOptimizer(FirstOrderOptimizer):
    ## Right now it's just implemented one gradient step with all the data
    
    def update_opt(self, loss, target, inputs, kl, extra_inputs=None, **kwargs):
        """
        :param inner_kl: Symbolic expression for inner kl
        :param outer_kl: Symbolic expression for outer kl
        :param meta_batch_size: number of MAML tasks, for batcher
        """
        super(PPOOptimizer, self).update_opt(loss, target, inputs, extra_inputs, **kwargs)
        if extra_inputs is None:
            extra_inputs = list()
        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, loss),
            f_kl=lambda: tensor_utils.compile_function(inputs + extra_inputs, kl),
        )

    def kl(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_kl"](*(tuple(inputs) + extra_inputs))

