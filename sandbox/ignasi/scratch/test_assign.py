import tensorflow as tf

def create_graph(inputs, outputs, W_ph):
    W = tf.Variable(initial_value=0., dtype=tf.float32, name="Weight")
    tf.assign(W, W_ph)
    loss = tf.reduce_mean(tf.square(outputs - W * inputs))


if __name__ == "__main__":
    pass