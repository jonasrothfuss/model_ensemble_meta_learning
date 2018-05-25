import time
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from sandbox.ignasi.samplers.batch_sampler import BatchSampler
from sandbox.ignasi.samplers.model_vectorized_sampler import ModelVectorizedSampler
from sandbox.ignasi.samplers.real_vectorized_sampler import VectorizedSampler
from rllab.sampler.utils import rollout
import numpy as np
from sandbox.ignasi.utils import DataBuffer


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            real_env,
            model_env,
            policy,
            baseline,
            dynamics_model,
            opt_model_itr=10,
            scope=None,
            n_itr=500,
            start_itr=0,
            num_paths=5,
            num_branches=10,
            max_path_length=1000,
            model_max_path_length=50,
            itr_to_collect_data=1,
            discount=1,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            param_noise=True,
            use_real_env=False,
            baseline_is_mean=True,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            samples_batch_size=5000,
            num_init_obs=1000,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        """
        self.env = self.real_env = real_env
        self._first = True
        self.use_real_env = use_real_env
        self.model_env = model_env
        self.policy = policy
        self.baseline = baseline
        self.dynamics_model = dynamics_model
        self.opt_model_itr = opt_model_itr
        self.scope = scope
        self.n_itr = n_itr
        self.opt_model_itr = opt_model_itr
        self.start_itr = start_itr
        self.batch_size = num_paths * max_path_length
        self.itr_to_collect_data = itr_to_collect_data
        self.num_paths = num_paths
        self.num_branches = num_branches
        self.max_path_length = max_path_length
        self.model_max_path_length = model_max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.real_world_data_collected = 0
        self.baseline_is_mean = baseline_is_mean
        assert num_init_obs <= samples_batch_size + self.batch_size
        self.num_init_obs = num_init_obs
        self.data_buffer = DataBuffer(dict(
            observations=np.zeros(real_env.observation_space.flat_dim, dtype=np.float32),
            actions=np.zeros(real_env.action_dim, dtype=np.float32),
            next_observations=np.zeros(real_env.observation_space.flat_dim, dtype=np.float32),
        ))

        if self.use_real_env:
            self.model_sampler = VectorizedSampler(self)
        else:
            self.model_sampler = ModelVectorizedSampler(self)
        self.real_sampler = BatchSampler(self) # TODO: I don't know if this is the correct thing to do
        self.init_opt()
        self._model_num = 0
        self.param_noise = param_noise
        self.samples_batch_size = samples_batch_size

    def start_worker(self):
        self.model_sampler.start_worker()
        self.real_sampler.start_worker()

    def shutdown_worker(self):
        self.model_sampler.shutdown_worker()
        self.real_sampler.shutdown_worker()

    def obtain_real_samples(self, itr, batch_size=None, multiplier=0):
        data = self.real_sampler.obtain_samples(itr, batch_size=batch_size)
        return data

    def obtain_model_samples(self, itr, observations):
        return self.model_sampler.obtain_samples(itr, observations)

    def process_real_samples(self, itr, paths, log=True):
        data = self.real_sampler.process_samples(itr, paths, log=log)
        self.data_buffer.add_data(dict((k, data[k]) for k in ('observations', 'next_observations', 'actions')))
        return data

    def process_model_samples(self, itr, paths):
        return self.model_sampler.process_samples(itr, paths)

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        self.initialize_unitialized_variables(sess)

        self.start_worker()
        start_time = time.time()
        prev_returns = [1e8]
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                if self.use_real_env or itr % self.itr_to_collect_data == 0:
                    logger.log("Obtaining real samples...")
                    real_paths = self.obtain_real_samples(itr)
                    real_samples = self.process_real_samples(itr, real_paths, log=True)
                    self.real_world_data_collected += len(real_samples['observations'])
                    new_return = np.mean(real_samples['undiscounted_returns'])
                    logger.log("New Return is {} and Previous Return was {}".format(new_return, np.mean(prev_returns)))
                    if not self.use_real_env and np.mean(prev_returns) > new_return:
                        self.optimize_model(itr)
                    if itr == 0:
                        prev_returns = [new_return]
                    elif itr != 0 and not self.use_real_env and np.mean(prev_returns) > new_return:
                        pass
                    else:
                        prev_returns = (prev_returns + [new_return])[-5:]
                else:
                    real_samples = self.process_real_samples(itr, real_paths, log=True)
                logger.log("Obtaining model samples...")
                init_obs = self.get_init_states(mode='last', size=self.num_init_obs)
                model_paths = self.obtain_model_samples(itr, init_obs)
                logger.log("Logging diagnostics...")
                self.log_diagnostics(real_paths)
                logger.log("Processing model samples...")
                samples_data = self.process_model_samples(itr, model_paths)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('RealWorldTS', self.real_world_data_collected)
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
        self.shutdown_worker()
        if created_session:
            sess.close()

    def log_diagnostics(self, paths):
        self.real_env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def optimize_model(self, itr):
        logger.log("Obtaining real samples for training the model...")
        if self._first:
            self.policy._random = True
            epochs = 100
            self._first = False
        else:
            epochs = 50
        if self.param_noise:
            pi_params = self.policy.get_param_values()
            self.policy.set_param_values(pi_params + 0.1 * (np.abs(pi_params) + 1e-4)* np.random.normal(size=pi_params.shape))
            real_paths = self.obtain_real_samples(itr, batch_size=self.samples_batch_size)
            self.policy.set_param_values(pi_params)
        else:
            real_paths = self.obtain_real_samples(itr, batch_size=self.samples_batch_size)
        real_samples = self.process_real_samples(itr, real_paths, log=False)
        self.real_world_data_collected += len(real_samples['observations'])
        data = self.data_buffer.get_data()
        logger.log("Fitting the model...")
        # self.dynamics_model.reinit_model()
        # self._model_num = (self._model_num + 1) % self.dynamics_model.num_models
        self.dynamics_model.fit(data['observations'], data['actions'], data['next_observations'], verbose=True, epochs=epochs)
        self.policy._random = False

    def get_init_states(self, mode='last', size=500):
        if mode == 'last':
            observations = self.data_buffer.get_data()['observations']
            init_obs = observations[-size:]
        elif mode == 'random':
            observations = self.data_buffer.get_data()['observations']
            idxs = np.random.randint(0, len(observations), size)
            init_obs = observations[idxs]
        else:
            raise NotImplementedError
        return init_obs



    def initialize_unitialized_variables(self, sess):
        uninit_variables = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_variables.append(var)
        sess.run(tf.variables_initializer(uninit_variables))