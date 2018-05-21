import matplotlib
matplotlib.use('Pdf')

import matplotlib.pyplot as plt
import os.path as osp
import rllab.misc.logger as logger
import rllab_maml.plotter as plotter
import tensorflow as tf
import time

from rllab_maml.algos.base import RLAlgorithm
from rllab_maml.sampler.stateful_pool import singleton_pool

from sandbox.jonas.sampler import RandomVectorizedSampler, MAMLModelVectorizedSampler, MAMLVectorizedSampler
from sandbox.jonas.sampler.MAML_sampler.maml_batch_sampler import BatchSampler


class ModelBatchMAMLPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with maml.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            dynamics_model,
            baseline,
            scope=None,
            n_itr=20,
            start_itr=0,
            # Note that the number of trajectories for grad upate = batch_size
            # Defaults are 10 trajectories of length 500 for gradient update
            batch_size_env_samples=10,
            batch_size_dynamics_samples=100,
            initial_random_samples=None,
            max_path_length=100,
            num_grad_updates=1,
            discount=0.99,
            gae_lambda=1,
            dynamic_model_epochs=(30, 10),
            num_maml_steps_per_iter=10,
            retrain_model_when_reward_decreases=True,
            reset_policy_std=False,
            reinit_model_cycle=0,
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
            use_maml=True,
            load_policy=None,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :param dynamics_model: Dynamics model ensemble
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size_env_samples: Number of policy rollouts for each model/policy
        :param batch_size_dynamics_samples: Number of (imaginary) policy rollouts with each dynamics model
        :param initial_random_samples: either None -> use initial policy to sample from env
                                       or int: number of random samples at first iteration to train dynamics model
                                               if provided, in the first iteration no samples from the env are generated
                                               with the policy
        :param max_path_length: Maximum length of a single rollout.
        :param meta_batch_size: Number of tasks sampled per meta-update
        :param num_grad_updates: Number of fast gradient updates
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param dynamic_model_epochs: (2-tuple) number of epochs to train the dynamics model
                                        (n_epochs_at_first_iter, n_epochs_after_first_iter)
        :param num_maml_steps_per_iter: number of policy gradients steps before retraining dynamics model
        :param retrain_model_when_reward_decreases: (boolean) if true - stop inner gradient steps when performance decreases
        :param reinit_model_cycle: number of iterations before re-initializing the dynamics model (if 0 the dynamic model is not re-initialized at all)
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.load_policy = load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr

        self.num_models = dynamics_model.num_models
        self.meta_batch_size = self.num_models # set meta_batch_size to number of dynamic models

        # batch_size is the number of trajectories for one fast grad update.
        self.batch_size = batch_size_env_samples * max_path_length * self.num_models # batch_size for env sampling
        self.batch_size_dynamics_samples = batch_size_dynamics_samples * max_path_length * self.num_models # batch_size for model sampling
        self.initial_random_samples = initial_random_samples

        # self.batch_size is the number of total transitions to collect.
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda

        self.dynamic_model_epochs = dynamic_model_epochs
        self.num_maml_steps_per_iter = num_maml_steps_per_iter
        self.retrain_model_when_reward_decreases = retrain_model_when_reward_decreases
        self.reset_policy_std = reset_policy_std
        self.reinit_model = reinit_model_cycle

        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.num_grad_updates = num_grad_updates # number of gradient steps during training

        ''' setup sampler classes '''

        # env sampler - get samples from environment using the policy
        if sampler_cls is None:
            if singleton_pool.n_parallel > 1:
                sampler_cls = BatchSampler
                sampler_args = dict(n_envs=self.meta_batch_size)
            else:
                sampler_cls = MAMLVectorizedSampler
                sampler_args = dict(n_tasks=self.meta_batch_size, n_envs=self.meta_batch_size * batch_size_env_samples)
        self.env_sampler = sampler_cls(self, **sampler_args)

        # model sampler - makes (imaginary) rollouts with the estimated dynamics model ensemble
        self.model_sampler = MAMLModelVectorizedSampler(self)

        # random sampler - (initially) collects random samples from the environment to train the dynamics model
        if self.initial_random_samples:
            self.random_sampler = RandomVectorizedSampler(self)
        else:
            self.random_sampler = None

    def start_worker(self):
        self.env_sampler.start_worker()
        self.model_sampler.start_worker()

        if self.initial_random_samples:
            self.random_sampler.start_worker()

        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.env_sampler.shutdown_worker()
        self.model_sampler.shutdown_worker()

    def obtain_env_samples(self, itr, reset_args=None, log_prefix=''):
        paths = self.env_sampler.obtain_samples(itr, reset_args, return_dict=True, log_prefix=log_prefix)
        assert type(paths) == dict
        return paths

    def obtain_model_samples(self, itr, log=False):
        return self.model_sampler.obtain_samples(itr, log=log, return_dict=True)

    def obtain_random_samples(self, itr, log=False):
        assert self.random_sampler is not None
        assert self.initial_random_samples is not None
        return self.random_sampler.obtain_samples(itr, num_samples=self.initial_random_samples, log=log,
                                                  log_prefix='EnvSampler-')

    def process_samples_for_dynamics(self, itr, paths):
        return self.model_sampler.process_samples(itr, paths, log=False)

    def process_samples_for_policy(self, itr, paths, log=True, log_prefix='DynTrajs-', return_reward=False):
        return self.env_sampler.process_samples(itr, paths, log=log, log_prefix=log_prefix, return_reward=return_reward)

    def train(self):
        # TODO - make this a util
        flatten_list = lambda l: [item for sublist in l for item in sublist]

        with tf.Session() as sess:
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            if self.load_policy is not None:
                import joblib
                self.policy = joblib.load(self.load_policy)['policy']
            self.init_opt()
            self.initialize_uninitialized_variables(sess)

            self.all_paths = []

            self.start_worker()
            start_time = time.time()
            n_env_timesteps = 0

            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):

                    logger.record_tabular("mean_inner_stepsize", self.policy.get_mean_step_size())

                    ''' sample environment configuration '''
                    env = self.env
                    while not ('sample_env_params' in dir(env) or 'sample_goals' in dir(env)):
                        env = env._wrapped_env
                    if 'sample_goals' in dir(env):
                        learner_env_params = env.sample_goals(self.meta_batch_size)
                    elif 'sample_env_params':
                        learner_env_params = env.sample_env_params(self.meta_batch_size)

                    ''' get rollouts from the environment'''

                    if self.initial_random_samples and itr == 0:
                        logger.log("Obtaining random samples from the environment...")
                        new_env_paths = self.obtain_random_samples(itr, log=True)

                        n_env_timesteps += self.initial_random_samples
                        logger.record_tabular("n_timesteps", n_env_timesteps)

                        self.all_paths.extend(new_env_paths)
                        samples_data_dynamics = self.random_sampler.process_samples(itr, self.all_paths,
                                                                                    log=True,
                                                                                    log_prefix='EnvTrajs-')  # must log in the same way as the model sampler below

                    else:
                        if self.reset_policy_std:
                            self.policy.set_std()
                        logger.log("Obtaining samples from the environment using the policy...")
                        new_env_paths = self.obtain_env_samples(itr, reset_args=learner_env_params,
                                                                log_prefix='EnvSampler-')
                        n_env_timesteps += self.batch_size
                        logger.record_tabular("n_timesteps", n_env_timesteps)

                        # flatten dict of paths per task/mode --> list of paths
                        new_env_paths = [path for task_paths in new_env_paths.values() for path in task_paths]
                        self.all_paths.extend(new_env_paths)
                        logger.log("Processing environment samples...")
                        # first processing just for logging purposes
                        self.model_sampler.process_samples(itr, new_env_paths, log=True, log_prefix='EnvTrajs-')

                        samples_data_dynamics = self.process_samples_for_dynamics(itr, self.all_paths)


                    ''' fit dynamics model '''

                    epochs = self.dynamic_model_epochs[min(itr, len(self.dynamic_model_epochs) - 1)]
                    if self.reinit_model and itr % self.reinit_model == 0:
                        self.dynamics_model.reinit_model()
                        epochs = self.dynamic_model_epochs[0] #todo: Probably to cycle through the dynamic_model_epochs
                    logger.log("Training dynamics model for %i epochs ..." % (epochs))
                    self.dynamics_model.fit(samples_data_dynamics['observations_dynamics'],
                                            samples_data_dynamics['actions_dynamics'],
                                            samples_data_dynamics['next_observations_dynamics'],
                                            epochs=epochs, verbose=True) #TODO set verbose False

                    ''' MAML steps '''
                    for maml_itr in range(self.num_maml_steps_per_iter):

                        self.policy.switch_to_init_dist()  # Switch to pre-update policy

                        all_samples_data_maml_iter, all_paths_maml_iter = [], []
                        for step in range(self.num_grad_updates + 1):

                            logger.log("MAML Step %i%s of %i - Obtaining samples from the dynamics model..." % (
                                maml_itr + 1, chr(97 + step), self.num_maml_steps_per_iter))

                            new_model_paths = self.obtain_model_samples(itr)
                            assert type(new_model_paths) == dict and len(new_model_paths) == self.num_models
                            all_paths_maml_iter.append(new_model_paths)

                            logger.log("Processing samples...")
                            samples_data = {}
                            for key in new_model_paths.keys():  # the keys are the tasks
                                # don't log because this will spam the consol with every task.
                                samples_data[key] = self.process_samples_for_policy(itr, new_model_paths[key], log=False)
                            all_samples_data_maml_iter.append(samples_data)

                            # for logging purposes
                            _, mean_reward = self.process_samples_for_policy(itr,
                                                                             flatten_list(new_model_paths.values()),
                                                                             log='reward',
                                                                             log_prefix="DynTrajs%i%s-" % (
                                                                                 maml_itr + 1, chr(97 + step)),
                                                                             return_reward=True)

                            if step < self.num_grad_updates:
                                logger.log("Computing policy updates...")
                                self.policy.compute_updated_dists(samples_data)

                        if maml_itr == 0:
                            prev_rolling_reward_mean = mean_reward
                            rolling_reward_mean = mean_reward
                        else:
                            prev_rolling_reward_mean = rolling_reward_mean
                            rolling_reward_mean = 0.7 * rolling_reward_mean + 0.3 * mean_reward


                        # stop gradient steps when mean_reward decreases
                        if self.retrain_model_when_reward_decreases and rolling_reward_mean < prev_rolling_reward_mean:
                            logger.log(
                                "Stopping policy gradients steps since rolling mean reward decreased from %.2f to %.2f" % (
                                    prev_rolling_reward_mean, rolling_reward_mean))
                            # complete some logging stuff
                            for i in range(maml_itr + 1, self.num_maml_steps_per_iter):
                                logger.record_tabular('DynTrajs%ia-AverageReturn' % i, None)
                                logger.record_tabular('DynTrajs%ib-AverageReturn' % i, None)
                            break

                        logger.log("MAML Step %i of %i - Optimizing policy..." % (maml_itr + 1, self.num_maml_steps_per_iter))
                        # This needs to take all samples_data so that it can construct graph for meta-optimization.
                        self.optimize_policy(itr, all_samples_data_maml_iter, log=False)


                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, all_samples_data_maml_iter[-1])  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = all_samples_data_maml_iter[-1]["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    logger.dump_tabular(with_prefix=False)


            self.shutdown_worker()

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
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

    def optimize_policy(self, itr, samples_data, log=True):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def initialize_uninitialized_variables(self, sess):
        uninit_vars = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        sess.run(tf.variables_initializer(uninit_vars))