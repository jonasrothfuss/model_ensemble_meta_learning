from sandbox.rocky.tf.core.network import MLP

import tensorflow as tf
import numpy as np
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import logger
from collections import OrderedDict
import sandbox.rocky.tf.core.layers as L
from sandbox.ours.dynamics import MLPDynamicsModel
import time


class PointEnvFakeModelEnsemble(Serializable):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self, env_spec, num_models=5, error_range_around_goal=0.5, bias_range=0.05, error_std=0.01, goal=(0,0),
                 error_at_goal=False, smooth_error=False, **kwargs):
        self.num_models = num_models
        self.env_spec = env_spec
        self.obs_space_dims = 2
        self.action_space_dims = 2
        self.error_range_around_goal = error_range_around_goal
        self.bias_range = bias_range
        self.error_std = error_std
        self.goal = np.asarray(goal)
        self.error_at_goal = error_at_goal
        self.smooth_error = smooth_error

        Serializable.quick_init(self, locals())

        self.model_biases = np.random.uniform(-self.bias_range, self.bias_range,
                                              size=(self.obs_space_dims, self.num_models))

    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True, valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False):
        """
        Fits the NN dynamics model
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after taking action - numpy array of shape (n_samples, ndim_obs)
        :param epochs: number of training epochs
        :param compute_normalization: boolean indicating whether normalization shall be (re-)computed given the data
        :param valid_split_ratio: relative size of validation split (float between 0.0 and 1.0)
        :param (boolean) whether to log training stats in tabular format
        :param verbose: logging verbosity
        """
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        self.model_biases = np.random.uniform(-self.bias_range, self.bias_range, size=(self.obs_space_dims, self.num_models))

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

        true_delta = np.tile(np.expand_dims(np.clip(act, -0.1, 0.1), 2), (1, 1, self.num_models)) # true point env delta

        delta_error = self._delta_error(obs)
        delta = true_delta + delta_error

        assert delta.ndim == 3

        pred_obs = obs_original[:, :, None] + delta

        batch_size = delta.shape[0]
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

        pred_obs_model_stacked = self.predict(obs_batches, act_batches, pred_type='all')
        pred_obs_batches = np.concatenate([pred_obs_split[:, :, i] for i, pred_obs_split in enumerate(np.vsplit(pred_obs_model_stacked, self.num_models))], axis=0)

        assert pred_obs_batches.shape == obs_batches.shape
        return pred_obs_batches

    def _delta_error(self, obs):

        if self.smooth_error:
            distances = np.linalg.norm(obs - self.goal[None, :], axis=1)
            normalized_distances = distances / np.max(distances)

            error_mask = (1-normalized_distances)**2 if self.error_at_goal else normalized_distances**2
        else:
            if self.error_at_goal:
                error_mask = (np.linalg.norm(obs-self.goal[None, :], axis=1) < self.error_range_around_goal).astype(np.float32)
            else:
                error_mask = (np.linalg.norm(obs - self.goal[None, :], axis=1) > self.error_range_around_goal).astype(
                    np.float32)
        error_mask = np.tile(error_mask.reshape((obs.shape[0], 1, 1)), (1, obs.shape[1],self.num_models))

        delta_error = np.random.normal(loc=self.model_biases, scale=self.error_std, size=obs.shape + (self.num_models,))

        # mask out delta error in certain regions
        delta_error = np.multiply(error_mask, delta_error)

        assert delta_error.shape == obs.shape + (self.num_models,)
        return delta_error

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
        pass

