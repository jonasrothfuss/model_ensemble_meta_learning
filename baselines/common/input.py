import tensorflow as tf
from gym.spaces import Discrete, Box
from rllab import spaces as rllab_spaces

def observation_input(ob_space, batch_size=None, name='Ob'):
    '''
    Build observation input with encoding depending on the 
    observation space type
    Params:
    
    ob_space: observation space (should be one of gym.spaces)
    batch_size: batch size for input (default is None, so that resulting input placeholder can take tensors with any batch size)
    name: tensorflow variable name for input placeholder

    returns: tuple (input_placeholder, processed_input_tensor)
    '''
    if isinstance(ob_space, Discrete) or isinstance(ob_space, rllab_spaces.Discrete):
        input_x  = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        processed_x = tf.to_float(tf.one_hot(input_x, ob_space.n))
        return input_x, processed_x

    elif isinstance(ob_space, Box) or isinstance(ob_space, rllab_spaces.Box):
        input_shape = (batch_size,) + ob_space.shape
        input_x = tf.placeholder(shape=input_shape, dtype=tf.float32, name=name)
        processed_x = tf.to_float(input_x)
        return input_x, processed_x

    else:
        raise NotImplementedError

 
