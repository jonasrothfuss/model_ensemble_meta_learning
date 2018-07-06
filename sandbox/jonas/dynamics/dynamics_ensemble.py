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
import time


class MLPDynamicsEnsemble(MLPDynamicsModel):
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
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99
                 ):

        Serializable.quick_init(self, locals())

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.batch_size = batch_size
        self.step_size = step_size
        self.num_models = num_models
        self.name = name

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env_spec.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env_spec.action_space.shape[0]

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
            self.delta_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

            # split stack into the batches for each model --> assume each model receives a batch of the same size
            self.obs_model_batches = tf.split(self.obs_model_batches_stack_ph, self.num_models, axis=0)
            self.act_model_batches = tf.split(self.act_model_batches_stack_ph, self.num_models, axis=0)
            self.delta_model_batches = tf.split(self.delta_model_batches_stack_ph, self.num_models, axis=0)

            # reuse previously created MLP but each model receives its own batch
            delta_preds = []
            self.obs_next_pred = []
            self.loss_model_batches = []
            self.train_op_model_batches = []
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
                loss = tf.reduce_mean((self.delta_model_batches[i] - mlp.output) ** 2)
                self.loss_model_batches.append(loss)
                self.train_op_model_batches.append(optimizer(self.step_size).minimize(loss))
            self.delta_pred_model_batches_stack = tf.concat(delta_preds, axis=0) # shape: (batch_size_per_model*num_models, ndim_obs)

            # tensor_utils
            self.f_delta_pred_model_batches = tensor_utils.compile_function([self.obs_model_batches_stack_ph, self.act_model_batches_stack_ph], self.delta_pred_model_batches_stack)

        LayersPowered.__init__(self, [mlp.output_layer for mlp in mlps])

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

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.get_default_session()

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(obs, act, obs_next)

        if self.normalize_input:
            # normalize data
            obs, act, delta = self._normalize_data(obs, act, obs_next)
            assert obs.ndim == act.ndim == obs_next.ndim == 2
        else:
            delta = obs_next - obs

        # split into valid and test set
        obs_train_batches = []
        act_train_batches = []
        delta_train_batches = []
        obs_test_batches = []
        act_test_batches = []
        delta_test_batches = []
        for i in range(self.num_models):
            obs_train, act_train, delta_train, obs_test, act_test, delta_test = train_test_split(obs, act, delta,
                                                                                             test_split_ratio=valid_split_ratio)
            obs_train_batches.append(obs_train)
            act_train_batches.append(act_train)
            delta_train_batches.append(delta_train)
            obs_test_batches.append(obs_test)
            act_test_batches.append(act_test)
            delta_test_batches.append(delta_test)
            # create data queue
        if self.next_batch is None:
            self.next_batch, self.iterator = self._data_input_fn(obs_train_batches, act_train_batches, delta_train_batches,
                                                       batch_size=self.batch_size)

        valid_loss_rolling_average = None
        train_op_to_do = self.train_op_model_batches
        idx_to_remove = []
        epoch_times = []
        epochs_per_model = []

        """ ------- Looping over training epochs ------- """
        for epoch in range(epochs):

            # initialize data queue
            feed_dict = dict(
                list(zip(self.obs_batches_dataset_ph, obs_train_batches)) +
                list(zip(self.act_batches_dataset_ph, act_train_batches)) +
                list(zip(self.delta_batches_dataset_ph, delta_train_batches))
            )
            sess.run(self.iterator.initializer, feed_dict=feed_dict)

            # preparations for recording training stats
            epoch_start_time = time.time()
            batch_losses = []

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            while True:
                try:
                    obs_act_delta = sess.run(self.next_batch)
                    obs_batch_stack = np.concatenate(obs_act_delta[:self.num_models], axis=0)
                    act_batch_stack = np.concatenate(obs_act_delta[self.num_models:2*self.num_models], axis=0)
                    delta_batch_stack = np.concatenate(obs_act_delta[2*self.num_models:], axis=0)

                    # run train op
                    batch_loss_train_ops= sess.run(self.loss_model_batches + train_op_to_do,
                                                   feed_dict={self.obs_model_batches_stack_ph: obs_batch_stack,
                                                              self.act_model_batches_stack_ph: act_batch_stack,
                                                              self.delta_model_batches_stack_ph: delta_batch_stack})

                    batch_loss = np.array(batch_loss_train_ops[:self.num_models])
                    batch_losses.append(batch_loss)

                except tf.errors.OutOfRangeError:
                    obs_test_stack = np.concatenate(obs_test_batches, axis=0)
                    act_test_stack = np.concatenate(act_test_batches, axis=0)
                    delta_test_stack = np.concatenate(delta_test_batches, axis=0)
                    # compute validation loss
                    valid_loss = sess.run(self.loss_model_batches,
                                          feed_dict={self.obs_model_batches_stack_ph: obs_test_stack,
                                                     self.act_model_batches_stack_ph: act_test_stack,
                                                     self.delta_model_batches_stack_ph: delta_test_stack})
                    valid_loss = np.array(valid_loss)
                    if valid_loss_rolling_average is None:
                        valid_loss_rolling_average = 1.5 * valid_loss  # set initial rolling to a higher value avoid too early stopping
                        valid_loss_rolling_average_prev = 2.0 * valid_loss
                        for i in range(len(valid_loss)):
                            if valid_loss[i] < 0:
                                valid_loss_rolling_average[i] = valid_loss[i]/1.5  # set initial rolling to a higher value avoid too early stopping
                                valid_loss_rolling_average_prev[i] = valid_loss[i]/2.0

                    valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average \
                                                 + (1.0-rolling_average_persitency)*valid_loss

                    if verbose:
                        str_mean_batch_losses = ' '.join(['%.4f'%x for x in np.mean(batch_losses, axis=0)])
                        str_valid_loss = ' '.join(['%.4f'%x for x in valid_loss])
                        str_valid_loss_rolling_averge = ' '.join(['%.4f'%x for x in valid_loss_rolling_average])
                        logger.log("Training NNDynamicsModel - finished epoch %i --"
                                   "train loss: %s  valid loss: %s  valid_loss_mov_avg: %s"
                                   %(epoch, str_mean_batch_losses, str_valid_loss, str_valid_loss_rolling_averge))
                    break

            for i in range(self.num_models):
                if (valid_loss_rolling_average_prev[i] < valid_loss_rolling_average[i] or epoch == self.epochs - 1) and i not in idx_to_remove:
                    idx_to_remove.append(i)
                    epochs_per_model.append(epoch)
                    logger.log('Stopping Training of Model %i since its valid_loss_rolling_average decreased'%i)

            train_op_to_do = [op for idx, op in enumerate(self.train_op_model_batches) if idx not in idx_to_remove]

            if not idx_to_remove: epoch_times.append(time.time() - epoch_start_time) # only track epoch times while all models are trained

            if not train_op_to_do:
                logger.log('Stopping DynamicsEnsemble Training since valid_loss_rolling_average decreased')
                break
            valid_loss_rolling_average_prev = valid_loss_rolling_average

        """ ------- Tabular Logging ------- """
        if log_tabular:
            logger.record_tabular('AvgModelEpochTime', np.mean(epoch_times))
            assert len(epochs_per_model) == self.num_models
            logger.record_tabular('AvgEpochsPerModel', np.mean(epochs_per_model))
            logger.record_tabular('StdEpochsPerModel', np.std(epochs_per_model))


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

        if self.normalize_input:
            obs_batches, act_batches = self._normalize_data(obs_batches, act_batches)
            delta_batches = np.array(self.f_delta_pred_model_batches(obs_batches, act_batches))
            delta_batches = denormalize(delta_batches, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta_batches = np.array(self.f_delta_pred(obs_batches, act_batches))

        assert delta_batches.ndim == 2

        pred_obs_batches = obs_batches_original + delta_batches
        assert pred_obs_batches.shape == obs_batches.shape
        return pred_obs_batches


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

    def _data_input_fn(self, obs_batches, act_batches, delta_batches, batch_size=500, buffer_size=100000):
        """ Takes in train data an creates an a symbolic nex_batch operator as well as an iterator object """

        assert len(obs_batches) == len(act_batches) == len(delta_batches)
        obs, act, delta = obs_batches[0], act_batches[0], delta_batches[0]
        assert obs.ndim == act.ndim == delta.ndim, "inputs must have 2 dims"
        assert obs.shape[0] == act.shape[0] == delta.shape[0], "inputs must have same length along axis 0"
        assert obs.shape[1] == delta.shape[1], "obs and obs_next must have same length along axis 1 "

        self.obs_batches_dataset_ph = [tf.placeholder(tf.float32, (None, obs.shape[1])) for _ in range(self.num_models)]
        self.act_batches_dataset_ph = [tf.placeholder(tf.float32, (None, act.shape[1])) for _ in range(self.num_models)]
        self.delta_batches_dataset_ph = [tf.placeholder(tf.float32, (None, delta.shape[1])) for _ in range(self.num_models)]

        dataset = tf.data.Dataset.from_tensor_slices(
            tuple(self.obs_batches_dataset_ph + self.act_batches_dataset_ph + self.delta_batches_dataset_ph)
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        return next_batch, iterator


def denormalize(data_array, mean, std):
    if data_array.ndim == 3: # assumed shape (batch_size, ndim_obs, n_models)
        return data_array * (std[None, :, None] + 1e-10) + mean[None, :, None]
    elif data_array.ndim == 2:
        return data_array * (std[None, :] + 1e-10) + mean[None, :]


def train_test_split(obs, act, delta, test_split_ratio=0.2):
    assert obs.shape[0] == act.shape[0] == delta.shape[0]
    dataset_size = obs.shape[0]
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(dataset_size * (1-test_split_ratio))

    idx_train = indices[:split_idx]
    idx_test = indices[split_idx:]
    assert len(idx_train) + len(idx_test) == dataset_size

    return obs[idx_train, :], act[idx_train, :], delta[idx_train, :], \
           obs[idx_test, :], act[idx_test, :], delta[idx_test, :]
