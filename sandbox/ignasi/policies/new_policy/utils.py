import numpy as np
from collections import OrderedDict
from rllab_maml.core.serializable import Serializable
from rllab.misc import logger
from rllab_maml.misc.tensor_utils import flatten_tensors, unflatten_tensors
from sandbox.ours.core.utils import make_dense_layer_with_bias_transform, forward_dense_bias_transform, \
    make_dense_layer
import tensorflow as tf
from sandbox_maml.rocky.tf.misc.xavier_init import xavier_initializer
from sandbox_maml.rocky.tf.core.utils import make_input, make_param_layer, forward_param_layer, forward_dense_layer
load_params = True


def create_MLP(name, input_shape, output_dim, hidden_sizes,
               hidden_W_init=xavier_initializer(), hidden_b_init=tf.zeros_initializer(),
               output_W_init=xavier_initializer(), output_b_init=tf.zeros_initializer(),
               weight_normalization=False, bias_transform=False, param_noise_std_ph=None):

    all_params = OrderedDict()

    cur_shape = input_shape
    with tf.variable_scope(name):
        if bias_transform:
            for idx, hidden_size in enumerate(hidden_sizes):
                # hidden layers
                W, b, bias_transform, cur_shape = make_dense_layer_with_bias_transform(
                    cur_shape,
                    num_units=hidden_size,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    bias_transform=hidden_b_init,
                    weight_norm=weight_normalization,
                )
                all_params['W' + str(idx)] = W
                all_params['b' + str(idx)] = b
                all_params['bias_transform' + str(idx)] = bias_transform

            # output layer
            W, b, bias_transform, _ = make_dense_layer_with_bias_transform(
                cur_shape,
                num_units=output_dim,
                name='output',
                W=hidden_W_init,
                b=hidden_b_init,
                bias_transform=hidden_b_init,
                weight_norm=weight_normalization,
                param_noise_std_ph=param_noise_std_ph
            )
            all_params['W' + str(len(hidden_sizes))] = W
            all_params['b' + str(len(hidden_sizes))] = b
            all_params['bias_transform' + str(len(hidden_sizes))] = bias_transform

        else:
            for idx, hidden_size in enumerate(hidden_sizes):
                W, b, cur_shape = make_dense_layer(
                    cur_shape,
                    num_units=hidden_size,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_norm=weight_normalization,
                )
                all_params['W' + str(idx)] = W
                all_params['b' + str(idx)] = b
            W, b, _ = make_dense_layer(
                cur_shape,
                num_units=output_dim,
                name='output',
                W=output_W_init,
                b=output_b_init,
                weight_norm=weight_normalization,
                param_noise_std_ph=param_noise_std_ph
            )
            all_params['W' + str(len(hidden_sizes))] = W
            all_params['b' + str(len(hidden_sizes))] = b

    return all_params


def forward_MLP(name, input_shape, n_hidden, hidden_nonlinearity, output_nonlinearity,
                all_params, input_tensor=None, batch_normalization=False, reuse=True,
                is_training=False, bias_transform=False):
    # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
    # set reuse to False if the first time this func is called.
    with tf.variable_scope(name):
        if input_tensor is None:
            l_in = make_input(shape=input_shape, input_var=None, name='input')
        else:
            l_in = input_tensor

        l_hid = l_in

        for idx in range(n_hidden):
            bias_transform_ = all_params['bias_transform' + str(idx)] if bias_transform else None
            l_hid = forward_dense_bias_transform(l_hid, all_params['W' + str(idx)], all_params['b' + str(idx)],
                                                 bias_transform=bias_transform_,
                                                 batch_norm=batch_normalization,
                                                 nonlinearity=hidden_nonlinearity,
                                                 scope=str(idx), reuse=reuse,
                                                 is_training=is_training
                                                 )

        bias_transform = all_params['bias_transform' + str(n_hidden)] if bias_transform else None
        output = forward_dense_bias_transform(l_hid, all_params['W' + str(n_hidden)],
                                              all_params['b' + str(n_hidden)],
                                              bias_transform=bias_transform, batch_norm=False,
                                              nonlinearity=output_nonlinearity,
                                              )
        return l_in, output
