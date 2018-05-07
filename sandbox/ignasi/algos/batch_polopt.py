import time
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.ignasi.samplers.model_vectorized_sampler import ModelVectorizedSampler
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
            num_paths=1,
            num_branches=2,
            max_path_length=500,
            model_max_path_length=200,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
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
        self.data_buffer = DataBuffer(dict(
            observations=np.zeros(real_env.observation_space.flat_dim, dtype=np.float32),
            actions=np.zeros(real_env.action_dim, dtype=np.float32),
            next_observations=np.zeros(real_env.observation_space.flat_dim, dtype=np.float32),
        ))

        self.model_sampler = ModelVectorizedSampler(self)
        self.real_sampler = BatchSampler(self) # TODO: I don't know if this is the correct thing to do
        self.init_opt()

    def start_worker(self):
        self.model_sampler.start_worker()
        self.real_sampler.start_worker()

    def shutdown_worker(self):
        self.model_sampler.shutdown_worker()
        self.real_sampler.shutdown_worker()

    def obtain_real_samples(self, itr):
        return self.real_sampler.obtain_samples(itr)

    def obtain_model_samples(self, itr, real_paths):
        return self.model_sampler.obtain_samples(itr, real_paths)

    def process_samples(self, itr, paths):
        return self.real_sampler.process_samples(itr, paths)

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        self.initialize_unitialized_variables(sess)

        self.start_worker()
        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                real_paths = self.obtain_real_samples(itr)
                model_paths = self.obtain_model_samples(itr, real_paths)
                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, real_paths, model_paths)
                logger.log("Logging diagnostics...")
                self.log_diagnostics(real_paths, model_paths)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                self.optimize_model(itr)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
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
        #TODO: Probably what I want here is to collect a lot of data and train the model at the beginning
        if itr and itr % self.opt_model_itr == 0:
            data = self.data_buffer.get_data()
            logger.log("Fitting the model...")
            self.dynamics_model.fit(data['observations'], data['actions'], data['next_observations'])

    def initialize_unitialized_variables(self, sess):
        uninit_variables = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_variables.append(var)

        sess.run(tf.variables_initializer(uninit_variables))