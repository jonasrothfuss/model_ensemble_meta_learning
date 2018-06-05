from sandbox.rocky.tf.core.network import MLP

import tensorflow as tf
import numpy as np
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import logger
from collections import OrderedDict
import sandbox.rocky.tf.core.layers as L
from sandbox.jonas.dynamics import MLPDynamicsModel


class BadDynamicsEnsemble(MLPDynamicsModel):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env_spec,
                 num_models=5,
                 hidden_sizes=(512, 512),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 batch_size=500,
                 step_size=0.001,
                 weight_normalization=False,
                 normalize_input=True,
                 optimizer=tf.train.AdamOptimizer,
                 gaussian_noise_output_std=0.0,
                 output_bias_range=0.0
                 ):

        Serializable.quick_init(self, locals())

        self.normalization = None
        self.normalize_input = normalize_input

        self.batch_size = batch_size
        self.step_size = step_size
        self.num_models = num_models
        self.name = name

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env_spec.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env_spec.action_space.shape[0]

        self.gaussian_noise_output_std = gaussian_noise_output_std
        self.output_bias_range = output_bias_range
        self.sample_output_bias()

        """ computation graph for training and simple inference """
        with tf.variable_scope(name):
            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

            # concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)

            # create MLP
            mlps = []
            delta_preds = []
            self.obs_next_pred = []
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i)):
                    mlp = MLP(name,
                              obs_space_dims,
                              hidden_sizes,
                              hidden_nonlinearity,
                              output_nonlinearity,
                              input_var=self.nn_input,
                              input_shape=(obs_space_dims+action_space_dims,),
                              weight_normalization=weight_normalization)
                    mlps.append(mlp)

                delta_preds.append(mlp.output)
            self.delta_pred = tf.stack(delta_preds, axis=2) # shape: (batch_size, ndim_obs, n_models)


            # define loss and train_op
            self.loss = tf.reduce_mean((self.delta_ph[:, :, None] - self.delta_pred)**2)
            self.optimizer = optimizer(self.step_size)
            self.train_op = self.optimizer.minimize(self.loss)

            # tensor_utils
            self.f_delta_pred = tensor_utils.compile_function([self.obs_ph, self.act_ph], self.delta_pred)

        """ computation graph for inference where each of the models receives a different batch"""
        with tf.variable_scope(name, reuse=True):
            # placeholders
            self.obs_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))

            # split stack into the batches for each model --> assume each model receives a batch of the same size
            self.obs_model_batches = tf.split(self.obs_model_batches_stack_ph, self.num_models, axis=0)
            self.act_model_batches = tf.split(self.act_model_batches_stack_ph, self.num_models, axis=0)

            # reuse previously created MLP but each model receives its own batch
            delta_preds = []
            self.obs_next_pred = []
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):
                    # concatenate action and observation --> NN input
                    nn_input = tf.concat([self.obs_model_batches[i], self.act_model_batches[i]], axis=1)
                    mlp = MLP(name,
                              obs_space_dims,
                              hidden_sizes,
                              hidden_nonlinearity,
                              output_nonlinearity,
                              input_var=nn_input,
                              input_shape=(obs_space_dims+action_space_dims,),
                              weight_normalization=weight_normalization)
                delta_preds.append(mlp.output)
            self.delta_pred_model_batches_stack = tf.concat(delta_preds, axis=0) # shape: (batch_size_per_model*num_models, ndim_obs)

            # tensor_utils
            self.f_delta_pred_model_batches = tensor_utils.compile_function([self.obs_model_batches_stack_ph, self.act_model_batches_stack_ph], self.delta_pred_model_batches_stack)

        LayersPowered.__init__(self, [mlp.output_layer for mlp in mlps])

    def predict(self, obs, act, pred_type='rand'):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param pred_type:  prediction type
                   - rand: choose one of the models randomly
                   - mean: mean prediction of all models
                   - all: returns the prediction of all the models
        :return: pred_obs_next: predicted batch of next observations -
                                shape:  (n_samples, ndim_obs) - in case of 'rand' and 'mean' mode
                                        (n_samples, ndim_obs, n_models) - in case of 'all' mode
        """
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            delta = np.array(self.f_delta_pred(obs, act))
            delta = denormalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta = np.array(self.f_delta_pred(obs, act))

        assert delta.ndim == 3

        pred_obs = obs_original[:, :, None] + delta
        batch_size = delta.shape[0]

        sampled_bias = np.stack([np.random.normal(loc=b, scale=self.gaussian_noise_output_std,
                                                        size=(batch_size, self.obs_space_dims)) for b in
                                       self.output_bias], axis=2)
        assert sampled_bias.shape == pred_obs.shape
        pred_obs += sampled_bias


        if pred_type == 'rand':
            # randomly selecting the prediction of one model in each row
            idx = np.random.randint(0, self.num_models, size=batch_size)
            pred_obs = np.stack([pred_obs[row, :, model_id] for row, model_id in enumerate(idx)], axis=0)
        elif pred_type == 'mean':
            pred_obs = np.mean(pred_obs, axis=2)
        elif pred_type == 'all':
            pass
        else:
            NotImplementedError('pred_type must be one of [rand, mean, all]')

        return pred_obs

    def predict_model_batches(self, obs_batches, act_batches):
        """
            Predict the batch of next observations for each model given the batch of current observations and actions for each model
            :param obs_batches: observation batches for each model concatenated along axis 0 - numpy array of shape (batch_size_per_model * num_models, ndim_obs)
            :param act_batches: action batches for each model concatenated along axis 0 - numpy array of shape (batch_size_per_model * num_models, ndim_act)
            :return: pred_obs_next_batch: predicted batch of next observations -
                                    shape:  (batch_size_per_model * num_models, ndim_obs)
        """
        assert obs_batches.shape[0] == act_batches.shape[0] and obs_batches.shape[0] % self.num_models == 0
        assert obs_batches.ndim == 2 and obs_batches.shape[1] == self.obs_space_dims
        assert act_batches.ndim == 2 and act_batches.shape[1] == self.action_space_dims

        obs_batches_original = obs_batches

        batch_size_per_model = obs_batches.shape[0] // self.num_models

        if self.normalize_input:
            obs_batches, act_batches = self._normalize_data(obs_batches, act_batches)
            delta_batches = np.array(self.f_delta_pred_model_batches(obs_batches, act_batches))
            delta_batches = denormalize(delta_batches, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta_batches = np.array(self.f_delta_pred(obs_batches, act_batches))

        assert delta_batches.ndim == 2

        pred_obs_batches = obs_batches_original + delta_batches
        assert pred_obs_batches.shape == obs_batches.shape

        sampled_bias = np.concatenate([np.random.normal(loc=b, scale=self.gaussian_noise_output_std,
                                                        size=(batch_size_per_model, self.obs_space_dims)) for b in self.output_bias], axis=0)
        assert sampled_bias.shape == pred_obs_batches.shape

        return pred_obs_batches + sampled_bias


    def predict_std(self, obs, act):
        """
        calculates the std of predicted next observations among the models
        given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :return: std_pred_obs: std of predicted next observatations - (n_samples, ndim_obs)
        """
        assert self.num_models > 1, "calculating the std requires at "
        pred_obs = self.predict(obs, act, pred_type='all')
        assert pred_obs.ndim == 3
        return np.std(pred_obs, axis=2)

    def reinit_model(self):
        sess = tf.get_default_session()
        if '_reinit_model_op' not in dir(self):
            self._reinit_model_op = [tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope=self.name+'/model_{}'.format(i))) for i in range(self.num_models)]
        sess.run(self._reinit_model_op)

    def sample_output_bias(self):
        self.output_bias = np.random.uniform(low=-self.output_bias_range, high=self.output_bias_range, size=self.num_models)

def denormalize(data_array, mean, std):
    if data_array.ndim == 3: # assumed shape (batch_size, ndim_obs, n_models)
        return data_array * (std[None, :, None] + 1e-10) + mean[None, :, None]
    elif data_array.ndim == 2:
        return data_array * (std[None, :] + 1e-10) + mean[None, :]