import numpy as np
from sandbox_maml.rocky.tf.core.utils import add_param
import tensorflow as tf
import sandbox_maml.rocky.tf.core.layers as L


def make_dense_layer(input_shape, num_units, name='fc', W=L.XavierUniformInitializer(), b=tf.zeros_initializer,
                     weight_norm=False, param_noise_std_ph=None, **kwargs):
    # make parameters
    num_inputs = int(np.prod(input_shape[1:]))
    W = add_param(W, (num_inputs, num_units), layer_name=name, name='W', weight_norm=weight_norm)
    if b is not None:
        b = add_param(b, (num_units,), layer_name=name, name='b', regularizable=False, weight_norm=weight_norm)
    output_shape = (input_shape[0], num_units)
    if param_noise_std_ph is not None:
        W_noised = W + tf.random_normal(W.get_shape(), stddev=param_noise_std_ph)
        b_noised = b + tf.random_normal(b.get_shape(), stddev=param_noise_std_ph)
        return W_noised, b_noised, output_shape
    else:
        return W, b, output_shape

def make_dense_layer_with_bias_transform(input_shape, num_units, name='fc', W=L.XavierUniformInitializer(), b=tf.zeros_initializer, bias_transform=tf.zeros_initializer, param_noise_std_ph=None, weight_norm=False, **kwargs):
    # make parameters
    num_inputs = int(np.prod(input_shape[1:]))

    # bias transform is half the size of input
    bias_transform = add_param(bias_transform, (num_inputs//2, ), layer_name=name, name='bias_transform', weight_norm=weight_norm)

    W = add_param(W, (num_inputs + num_inputs//2, num_units), layer_name=name, name='W', weight_norm=weight_norm)
    if b is not None:
        b = add_param(b, (num_units,), layer_name=name, name='b', regularizable=False, weight_norm=weight_norm)
    output_shape = (input_shape[0], num_units)
    if param_noise_std_ph is not None: # add gaussian noise to the parameters
        W_noised = W + tf.random_normal(W.get_shape(), stddev=param_noise_std_ph)
        b_noised = b + tf.random_normal(b.get_shape(), stddev=param_noise_std_ph)
        bias_transform_noised = bias_transform + tf.random_normal(bias_transform.get_shape(), stddev=param_noise_std_ph)
        return W_noised, b_noised, bias_transform_noised, output_shape
    else:
        return W, b, bias_transform, output_shape


def forward_dense_bias_transform(input, W, b, bias_transform=None, nonlinearity=tf.identity, batch_norm=False, scope='', reuse=True, is_training=False,):
    # compute output tensor
    if input.get_shape().ndims > 2:
        # if the input has more than two dimensions, flatten it into a
        # batch of feature vectors.
        input = tf.reshape(input, tf.stack([tf.shape(input)[0], -1]))

    if bias_transform is not None: # concatenate bias transform to input
        batch_size = tf.shape(input)[0]
        bias_transform = tf.expand_dims(bias_transform, 0) # make 2d so that it can be tiled
        tiled_bias_transform = tf.tile(bias_transform, (batch_size, 1))
        assert tiled_bias_transform.get_shape().ndims == 2

        input = tf.concat([input, tiled_bias_transform], axis=-1)

    activation = tf.matmul(input, W)
    if b is not None:
        activation = activation + tf.expand_dims(b, 0)

    if batch_norm:
        raise NotImplementedError('not supported')
    else:
        return nonlinearity(activation)
