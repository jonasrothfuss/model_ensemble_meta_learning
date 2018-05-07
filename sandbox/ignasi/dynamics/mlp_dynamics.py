from sandbox.rocky.tf.core.network import MLP

import tensorflow as tf
import numpy as np
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import logger
import sandbox.rocky.tf.core.layers as L
import joblib


class MLPDynamicsModel(LayersPowered, Serializable):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 num_models=1,
                 hidden_sizes=(512, 512),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 batch_size=200,
                 step_size=0.01,
                 batch_normalization=True,
                 weight_normalization=False,
                 normalize_inp=True,
                 normalize_out=True,
                 ):

        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            self.batch_size = batch_size
            self.step_size = step_size
            self.num_models = num_models

            # determine dimensionality of state and action space
            obs_space_dims = env.observation_space.shape[0]
            action_space_dims = env.action_space.shape[0]

            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.obs_next_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

            self.delta_target = self.obs_next_ph - self.obs_ph
            # concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)

            # TODO: Create the normlazation variables

            # create MLP
            mlps = []
            self.delta_pred = []
            self.obs_next_pred = []
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i)):
                    mlp = MLP(name,
                              obs_space_dims,
                              hidden_sizes,
                              hidden_nonlinearity,
                              output_nonlinearity,
                              input_var=self.nn_input,
                              input_shape = (obs_space_dims+action_space_dims,),
                              batch_normalization=batch_normalization,
                              weight_normalization=weight_normalization)
                    mlps.append(mlp)

                self.delta_pred.append(mlp.output)
            self.obs_next_pred.append(self.obs_ph + self.delta_pred)

            # define loss and train_op
            self.loss = tf.reduce_mean([tf.nn.l2_loss(self.delta_target - delta_pred)
                                        for delta_pred in self.delta_pred])
            self.train_op = tf.train.AdamOptimizer(self.step_size).minimize(self.loss)

            # tensor_utils
            self.f_next_obs_pred = tensor_utils.compile_function([self.obs_ph, self.act_ph], self.obs_next_pred)

        LayersPowered.__init__(self, [mlp.output_layer for mlp in mlps])

    def fit(self, obs, act, obs_next, epochs=100, verbose=False):
        """
        Fits the NN dynamics model
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after takeing action - numpy array of shape (n_samples, ndim_obs)
        :param epochs: number of training epochs
        :param verbose: logging verbosity
        """
        sess = tf.get_default_session()

        # create data queue
        next_batch, iterator = self._data_input_fn(obs, act, obs_next, batch_size=self.batch_size)

        # Training loop
        for epoch in range(epochs):

            # initialize data queue
            sess.run(iterator.initializer,
                          feed_dict={self.obs_dataset_ph: obs,
                                     self.act_dataset_ph: act,
                                     self.next_obs_dataset_ph: obs_next})

            batch_losses = []

            while True:
                try:
                    obs_batch, act_batch, obs_next_batch = sess.run(next_batch)
                    batch_loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.obs_ph: obs_batch,
                                                                         self.act_ph: act_batch,
                                                                         self.obs_next_ph: obs_next_batch})

                    batch_losses.append(batch_loss)

                except tf.errors.OutOfRangeError:
                    if verbose:
                        logger.log("Training NNDynamicsModel - finished epoch {} -- mean loss: {}".format(epoch, np.mean(batch_losses)))
                    break

    def predict(self, obs, act, pred_type='rand'):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :return: pred_obs_next: predicted batch of next observations
        """
        assert obs.ndim == 2 and act.ndim == 2, "inputs must have two dimensions"
        assert obs.shape[0] == act.shape[0]

        pred = np.array(self.f_next_obs_pred(obs, act))
        batch_size = obs.shape[0]

        if pred_type == 'rand':
            idx = np.random.randint(0, self.num_models, size=batch_size)
            pred = pred[idx, range(batch_size)]
        elif pred_type == 'mean':
            pred = np.mean(pred, axis=0)
        elif pred_type == 'all':
            pass
        return pred

    def _data_input_fn(self, obs, act, obs_next, batch_size=500, buffer_size=100000):
        """ Takes in train data an creates an a symbolic nex_batch operator as well as an iterator object """

        assert obs.ndim == act.ndim == obs_next.ndim, "inputs must have 2 dims"
        assert obs.shape[0] == act.shape[0] == obs_next.shape[0], "inputs must have same length along axis 0"
        assert obs.shape[1] == obs_next.shape[1], "obs and obs_next must have same length along axis 1 "

        self.obs_dataset_ph = tf.placeholder(tf.float32, obs.shape)
        self.act_dataset_ph = tf.placeholder(tf.float32, act.shape)
        self.next_obs_dataset_ph = tf.placeholder(tf.float32, obs_next.shape)

        dataset = tf.data.Dataset.from_tensor_slices((self.obs_dataset_ph,
                                                      self.act_dataset_ph,
                                                      self.next_obs_dataset_ph))
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        return next_batch, iterator
