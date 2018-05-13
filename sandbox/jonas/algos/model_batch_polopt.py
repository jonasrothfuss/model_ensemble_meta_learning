import time
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from rllab.sampler.utils import rollout

from sandbox.jonas.sampler import ModelVectorizedSampler, RandomVectorizedSampler, EnvVectorizedSampler


class ModelBatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            dynamics_model,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size_env_samples=5000,
            batch_size_dynamics_samples=40000,
            initial_random_samples=None,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            dynamic_model_epochs=(30, 10),
            num_gradient_steps_per_iter=10,
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
        :param dynamics_model: Dynamics Model
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size_env_samples: Number of samples from the environment per iteration.
        :param batch_size_dynamics_samples: Number of (imaginary) samples from the dynamics model
        :param initial_random_samples: either None -> use initial policy to sample from env
                                       or int: number of random samples at first iteration to train dynamics model
                                               if provided, in the first iteration no samples from the env are generated
                                               with the policy
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param dynamic_model_epochs: (2-tuple) number of epochs to train the dynamics model
                                        (n_epochs_at_first_iter, n_epochs_after_first_iter)
        :param num_gradient_steps_per_iter: number of policy gradients steps before retraining dynamics model
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        """
        self.env = env
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size_env_samples
        self.batch_size_dynamics_samples = batch_size_dynamics_samples
        self.initial_random_samples = initial_random_samples
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.dynamic_model_epochs = dynamic_model_epochs
        self.num_gradient_steps_per_iter = num_gradient_steps_per_iter
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon

        # sampler for the environment
        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = EnvVectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.env_sampler = sampler_cls(self, **sampler_args)

        # sampler for (imaginary) rollouts with the estimated dynamics model
        self.model_sampler = ModelVectorizedSampler(self)

        if self.initial_random_samples:
            self.random_sampler = RandomVectorizedSampler(self)
        else:
            self.random_sampler = None

        self.init_opt()

    def start_worker(self):
        self.env_sampler.start_worker()
        self.model_sampler.start_worker()

        if self.initial_random_samples:
            self.random_sampler.start_worker()

    def shutdown_worker(self):
        self.env_sampler.shutdown_worker()
        self.model_sampler.shutdown_worker()

    def obtain_env_samples(self, itr):
        return self.env_sampler.obtain_samples(itr, log_prefix='EnvSampler-')

    def obtain_model_samples(self, itr):
        return self.model_sampler.obtain_samples(itr)

    def obtain_random_samples(self, itr):
        assert self.random_sampler is not None
        assert self.initial_random_samples is not None
        return self.random_sampler.obtain_samples(itr, num_samples=self.initial_random_samples)

    def process_samples_for_dynamics(self, itr, paths):
        return self.model_sampler.process_samples(itr, paths, log=False)

    def process_samples_for_policy(self, itr, paths, log=True, log_prefix='DynTrajs-'):
        return self.env_sampler.process_samples(itr, paths, log=log, log_prefix=log_prefix)

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        self.initialize_unitialized_variables(sess)

        self.all_paths = []

        self.start_worker()
        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()

            with logger.prefix('itr #%d | ' % itr):


                # get rollouts from the env

                if self.initial_random_samples and itr == 0:
                    logger.log("Obtaining random samples from the environment...")
                    new_env_paths = self.obtain_random_samples(itr)
                    self.all_paths.extend(new_env_paths)
                    samples_data_dynamics = self.random_sampler.process_samples(itr, self.all_paths, log=True, log_prefix='EnvTrajs-') # must log in the same way as the model sampler below
                else:
                    logger.log("Obtaining samples from the environment using the policy...")
                    new_env_paths = self.obtain_env_samples(itr)
                    self.all_paths.extend(new_env_paths)
                    logger.log("Processing environment samples...")
                    # first processing just for logging purposes
                    self.model_sampler.process_samples(itr, new_env_paths, log=True, log_prefix='EnvTrajs-')

                    samples_data_dynamics = self.process_samples_for_dynamics(itr, self.all_paths)

                # fit dynamics model
                epochs = self.dynamic_model_epochs[min(itr, len(self.dynamic_model_epochs) - 1)]
                logger.log("Training dynamics model for %i epochs ..." % (epochs))
                self.dynamics_model.fit(samples_data_dynamics['observations_dynamics'],
                                        samples_data_dynamics['actions_dynamics'],
                                        samples_data_dynamics['next_observations_dynamics'],
                                        epochs=epochs)

                for gradient_itr in range(self.num_gradient_steps_per_iter):
                    # get imaginary rollouts
                    logger.log("Policy Gradient Step %i of %i - Obtaining samples from the dynamics model..."%(gradient_itr, self.num_gradient_steps_per_iter))
                    new_model_paths = self.obtain_model_samples(itr)

                    logger.log("Policy Gradient Step %i of %i - Processing dynamics model samples..."%(gradient_itr, self.num_gradient_steps_per_iter))
                    samples_data_model = self.process_samples_for_policy(itr, new_model_paths, log='reward', log_prefix='%i-DynTrajs-'%gradient_itr)

                    # logger.log("Policy Gradient Step %i of %i - Logging diagnostics..."%(gradient_itr, self.num_gradient_steps_per_iter))
                    # self.log_diagnostics(new_model_paths)

                    logger.log("Policy Gradient Step %i of %i - Optimizing policy..."%(gradient_itr, self.num_gradient_steps_per_iter))
                    self.optimize_policy(itr, samples_data_model, log=False)


                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data_model)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data_model["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    rollout(self.env, self.policy, animated=True, max_path_length=self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()
        if created_session:
            sess.close()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
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

    def optimize_policy(self, itr, samples_data, log=True, log_prefix=''):
        raise NotImplementedError

    def initialize_unitialized_variables(self, sess):
        uninit_variables = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_variables.append(var)

        sess.run(tf.variables_initializer(uninit_variables))