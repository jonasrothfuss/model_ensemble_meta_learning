import rllab.misc.logger as logger
import tensorflow as tf
import time
import numpy as np
import os
import joblib

from rllab_maml.algos.base import RLAlgorithm
from rllab_maml.sampler.stateful_pool import singleton_pool

from sandbox.ours.sampler import RandomVectorizedSampler, MAMLModelVectorizedSampler, MAMLVectorizedSampler
from sandbox.ours.sampler.MAML_sampler.maml_batch_sampler import BatchSampler

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
            meta_batch_size=None,
            initial_random_samples=None,
            max_path_length_env=100,
            max_path_length_dyn=None,
            num_grad_updates=1,
            discount=0.99,
            entropy_bonus=0,
            gae_lambda=1,
            dynamic_model_max_epochs=(1000, 1000),
            num_maml_steps_per_iter=10,
            reset_from_env_traj=False,
            dynamics_data_buffer_size=1e5,
            retrain_model_when_reward_decreases=True,
            reset_policy_std=False,
            reinit_model_cycle=0,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            sampler_cls=None,
            sampler_args=None,
            load_policy=None,
            frac_gpu=0.85,
            log_real_performance=True,
            clip_obs=False,
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
        :param meta_batch_size: Number of meta-tasks (default - meta_batch_size-dynamics_model.num_models)
        :param initial_random_samples: either None -> use initial policy to sample from env
                                       or int: number of random samples at first iteration to train dynamics model
                                               if provided, in the first iteration no samples from the env are generated
                                               with the policy
        :param max_path_length_env: Maximum length of a single rollout in the environment
        :param max_path_length_dyn: Maximum path length of a single (imaginary) rollout with the dynamics model
        :param num_grad_updates: Number of fast gradient updates
        :param discount: Discount.
        :param entropy_bonus_coef: Entropy bonus
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param dynamic_model_max_epochs: (int) maximum number of epochs for training the dynamics model
        :param num_maml_steps_per_iter: number of policy gradients steps before retraining dynamics model
        :param reset_from_env_traj: (boolean) whether to use the real environment observations for resetting the imaginary dynamics model rollouts
        :param dynamics_data_buffer_size: (int) size of the queue/buffer that holds data for the model training
        :param retrain_model_when_reward_decreases: (boolean) if true - stop inner gradient steps when performance decreases
        :param reset_policy_std: whether to reset the policy std after each iteration
        :param reinit_model_cycle: number of iterations before re-initializing the dynamics model (if 0 the dynamic model is not re-initialized at all)
        :param store_paths: Whether to save all paths data to the snapshot.
        :param frac_gpu: memory fraction of the gpu that shall be used for this task
        :param log_real_performance: (boolean) if true the pre-update and post-update performance in the real env is evaluated and logged
        :param clip_obs: (boolean) whether to clip the predicted next observations of the dynamics model in order to avoid numerical instabilities
        """
        self.env = env
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.load_policy = load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr

        # meta batch size and number of dynamics models
        self.num_models = dynamics_model.num_models
        if meta_batch_size is None:
            self.meta_batch_size = self.num_models # set meta_batch_size to number of dynamic models
        else:
            assert meta_batch_size % self.num_models == 0, "meta_batch_size must a multiple the number of models in the dynamics ensemble"
            self.meta_batch_size = meta_batch_size

        self.max_path_length = max_path_length_env
        self.max_path_length_dyn = max_path_length_dyn if max_path_length_dyn is not None else max_path_length_env

        # batch_size is the number of trajectories for one fast grad update.
        self.batch_size = batch_size_env_samples * max_path_length_env * self.meta_batch_size # batch_size for env sampling
        self.batch_size_dynamics_samples = batch_size_dynamics_samples * self.max_path_length_dyn * self.meta_batch_size # batch_size for model sampling
        if initial_random_samples is None:
            self.initial_random_samples = self.batch_size
        else:
            self.initial_random_samples = initial_random_samples
        self.discount = discount
        self.entropy_bonus = entropy_bonus
        self.gae_lambda = gae_lambda

        # dynamics model config
        self.dynamic_model_epochs = dynamic_model_max_epochs
        self.num_maml_steps_per_iter = num_maml_steps_per_iter
        self.reset_from_env_traj = reset_from_env_traj
        self.dynamics_data_buffer_size = dynamics_data_buffer_size
        self.retrain_model_when_reward_decreases = retrain_model_when_reward_decreases
        self.reset_policy_std = reset_policy_std
        self.reinit_model = reinit_model_cycle
        self.log_real_performance = log_real_performance

        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.num_grad_updates = num_grad_updates # number of gradient steps during training
        self.frac_gpu = frac_gpu

        ''' setup sampler classes '''

        # env sampler - get samples from environment using the policy
        if sampler_cls is None:
            sampler_cls = MAMLVectorizedSampler
            sampler_args = dict(n_tasks=self.meta_batch_size, n_envs=self.meta_batch_size * batch_size_env_samples, parallel=False)
        self.env_sampler = sampler_cls(self, **sampler_args)

        # model sampler - makes (imaginary) rollouts with the estimated dynamics model ensemble
        self.model_sampler = MAMLModelVectorizedSampler(self, max_path_length=max_path_length_dyn, clip_obs=clip_obs)

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

    def shutdown_worker(self):
        self.env_sampler.shutdown_worker()
        self.model_sampler.shutdown_worker()

    def obtain_env_samples(self, itr, reset_args=None, log_prefix=''):
        paths = self.env_sampler.obtain_samples(itr, reset_args, return_dict=True, log_prefix=log_prefix)
        assert type(paths) == dict
        return paths

    def obtain_model_samples(self, itr, log=False, traj_starting_obs=None, traj_starting_ts=None):
        return self.model_sampler.obtain_samples(itr, log=log, return_dict=True, traj_starting_obs=traj_starting_obs,
                                                 traj_starting_ts=traj_starting_ts)

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

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.frac_gpu

        with tf.Session(config=config) as sess:
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            if self.load_policy is not None:
                self.policy = joblib.load(self.load_policy)['policy']
            self.init_opt()
            self.initialize_uninitialized_variables(sess)

            self.all_paths = []

            self.start_worker()
            start_time = time.time()
            n_env_timesteps = 0

            """ ----- prepare stuff for kl heatplots --------- """

            resolution = 50
            linspace = np.linspace(-1.8, 1.8, resolution)
            x_points, y_points =  np.meshgrid(linspace, linspace)
            obs_grid_points = np.stack([x_points.flatten(), y_points.flatten()], axis=1)
            assert obs_grid_points.shape == (resolution**2, 2)

            if logger._snapshot_dir:
                DUMP_DIR = logger._snapshot_dir
            else:
                DUMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')

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

                    time_env_sampling_start = time.time()

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
                            logger.log("Resetting policy std")
                            self.policy.set_std()
                        logger.log("Obtaining samples from the environment using the policy...")
                        new_env_paths = self.obtain_env_samples(itr, reset_args=learner_env_params,
                                                                log_prefix='EnvSampler-')
                        n_env_timesteps += self.batch_size
                        logger.record_tabular("n_timesteps", n_env_timesteps)

                        # flatten dict of paths per task/mode --> list of paths
                        new_env_paths = [path for task_paths in new_env_paths.values() for path in task_paths]
                        # self.all_paths.extend(new_env_paths)
                        logger.log("Processing environment samples...")
                        # first processing just for logging purposes
                        self.model_sampler.process_samples(itr, new_env_paths, log=True, log_prefix='EnvTrajs-')

                        new_samples_data_dynamics = self.process_samples_for_dynamics(itr, new_env_paths)
                        for k, v in samples_data_dynamics.items():
                            samples_data_dynamics[k] = np.concatenate([v, new_samples_data_dynamics[k]], axis=0)[-int(self.dynamics_data_buffer_size):]

                    logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)

                    epochs = self.dynamic_model_epochs[min(itr, len(self.dynamic_model_epochs) - 1)]
                    logger.log("Training dynamics model for %i epochs ..." % (epochs))
                    self.dynamics_model.fit(samples_data_dynamics['observations_dynamics'],
                                            samples_data_dynamics['actions_dynamics'],
                                            samples_data_dynamics['next_observations_dynamics'],
                                            epochs=epochs, verbose=True)

                    ''' ------------- Making Plots ------------------  '''

                    logger.log("Evaluating the performance of the real policy")
                    self.policy.switch_to_init_dist()
                    env_paths_pre = self.obtain_env_samples(itr, reset_args=learner_env_params,
                                                            log_prefix='PrePolicy-')
                    samples_data = {}
                    for key in env_paths_pre.keys():
                        samples_data[key] = self.process_samples_for_policy(itr, env_paths_pre[key], log=False)
                    _ = self.process_samples_for_policy(itr, flatten_list(env_paths_pre.values()), log_prefix='PrePolicy-')

                    _ , agent_infos_pre = self.policy.get_actions_batch([obs_grid_points for _ in range(self.meta_batch_size)])

                    self.policy.compute_updated_dists(samples_data)
                    env_paths_post = self.obtain_env_samples(itr, reset_args=learner_env_params, log_prefix='PostPolicy-',)
                    _ = self.process_samples_for_policy(itr,  flatten_list(env_paths_post.values()), log_prefix='PostPolicy-')

                    _, agent_infos_post = self.policy.get_actions_batch(
                        [obs_grid_points for _ in range(self.meta_batch_size)])

                    # compute KL divergence between pre and post update policy
                    kl_pre_post = self.policy.distribution.kl(agent_infos_pre, agent_infos_post)
                    kl_pre_post_grid = kl_pre_post.reshape((self.meta_batch_size, resolution**2)).mean(axis=0).reshape((resolution,resolution))

                    model_std_grid = self.dynamics_model.predict_std(obs_grid_points, - 0.05 * obs_grid_points).mean(axis=1).reshape((resolution,resolution))

                    import matplotlib
                    matplotlib.use('agg')
                    import matplotlib.pyplot as plt
                    plt.style.use('ggplot')

                    img_filename = os.path.join(DUMP_DIR, 'kl_vs_model_std_plot_iter_%i' % itr)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    fig.tight_layout(pad=3)

                    ax1.set_title('KL-divergence')
                    ax1.set_ylabel('y')
                    ax1.set_xlabel('x')

                    # determine range of plot
                    point_env = self.env._wrapped_env._wrapped_env
                    env_center = (point_env.init_sampling_boundaries[0] + point_env.init_sampling_boundaries[1]) / 2
                    distance = np.abs(
                        point_env.init_sampling_boundaries[0] - point_env.init_sampling_boundaries[1]) / 2
                    extent = (env_center - 0.9*distance, env_center + 0.9*distance, env_center - 0.9*distance, env_center + 0.9*distance)

                    im1 = ax1.imshow(kl_pre_post_grid, extent=extent)
                    fig.colorbar(ax=ax1, mappable=im1, shrink=0.8, orientation='vertical')
                    ax1.grid(False)

                    ax2.set_title('Ensemble variance')
                    ax2.set_ylabel('y')
                    ax2.set_xlabel('x')
                    im2 = ax2.imshow(model_std_grid, extent=extent)
                    fig.colorbar(ax=ax2, mappable=im2, shrink=0.8, orientation='vertical')
                    ax2.grid(False)

                    # save plot
                    fig.savefig(img_filename)

                    # save plot data
                    plot_data={
                        'kl': kl_pre_post_grid,
                        'std': model_std_grid,
                        'extent': extent
                    }

                    plot_data_file = os.path.join(DUMP_DIR, 'kl_vs_model_std_plot_iter_%i.pkl' % itr)
                    joblib.dump(plot_data, plot_data_file)


                    ''' --------------- fit dynamics model --------------- '''

                    time_fit_start = time.time()

                    epochs = self.dynamic_model_epochs[min(itr, len(self.dynamic_model_epochs) - 1)]
                    if self.reinit_model and itr % self.reinit_model == 0:
                        self.dynamics_model.reinit_model()
                        epochs = self.dynamic_model_epochs[0]
                    logger.log("Training dynamics model for %i epochs ..." % (epochs))
                    self.dynamics_model.fit(samples_data_dynamics['observations_dynamics'],
                                            samples_data_dynamics['actions_dynamics'],
                                            samples_data_dynamics['next_observations_dynamics'],
                                            epochs=epochs, verbose=True, log_tabular=True)

                    logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)

                    ''' --------------- MAML steps --------------- '''

                    times_dyn_sampling = []
                    times_dyn_sample_processing = []
                    times_inner_step = []
                    times_outer_step = []

                    time_maml_steps_start = time.time()

                    for maml_itr in range(self.num_maml_steps_per_iter):

                        self.policy.switch_to_init_dist()  # Switch to pre-update policy

                        all_samples_data_maml_iter, all_paths_maml_iter = [], []
                        for step in range(self.num_grad_updates + 1):

                            ''' --------------- Sampling from Dynamics Models --------------- '''

                            logger.log("MAML Step %i%s of %i - Obtaining samples from the dynamics model..." % (
                                maml_itr + 1, chr(97 + step), self.num_maml_steps_per_iter))

                            time_dyn_sampling_start = time.time()

                            if self.reset_from_env_traj:
                                new_model_paths = self.obtain_model_samples(itr, traj_starting_obs=samples_data_dynamics['observations_dynamics'],
                                                                            traj_starting_ts=samples_data_dynamics['timesteps_dynamics'])
                            else:
                                new_model_paths = self.obtain_model_samples(itr)

                            assert type(new_model_paths) == dict and len(new_model_paths) == self.meta_batch_size
                            all_paths_maml_iter.append(new_model_paths)

                            times_dyn_sampling.append(time.time() - time_dyn_sampling_start)

                            ''' --------------- Processing Dynamics Samples --------------- '''

                            logger.log("Processing samples...")
                            time_dyn_sample_processing_start = time.time()
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

                            times_dyn_sample_processing.append(time.time() - time_dyn_sample_processing_start)

                            ''' --------------- Inner Policy Update --------------- '''

                            time_inner_step_start = time.time()

                            if step < self.num_grad_updates:
                                logger.log("Computing policy updates...")
                                self.policy.compute_updated_dists(samples_data)

                            times_inner_step.append(time.time() - time_inner_step_start)

                        if maml_itr == 0:
                            prev_rolling_reward_mean = mean_reward
                            rolling_reward_mean = mean_reward
                        else:
                            prev_rolling_reward_mean = rolling_reward_mean
                            rolling_reward_mean = 0.8 * rolling_reward_mean + 0.2 * mean_reward


                        # stop gradient steps when mean_reward decreases
                        if self.retrain_model_when_reward_decreases and rolling_reward_mean < prev_rolling_reward_mean:
                            logger.log(
                                "Stopping policy gradients steps since rolling mean reward decreased from %.2f to %.2f" % (
                                    prev_rolling_reward_mean, rolling_reward_mean))
                            # complete some logging stuff
                            for i in range(maml_itr + 1, self.num_maml_steps_per_iter):
                                logger.record_tabular('DynTrajs%ia-AverageReturn' % (i+1), 0.0)
                                logger.record_tabular('DynTrajs%ib-AverageReturn' % (i+1), 0.0)
                            break

                        ''' --------------- Meta Policy Update --------------- '''

                        logger.log("MAML Step %i of %i - Optimizing policy..." % (maml_itr + 1, self.num_maml_steps_per_iter))
                        time_outer_step_start = time.time()

                        # This needs to take all samples_data so that it can construct graph for meta-optimization.
                        self.optimize_policy(itr, all_samples_data_maml_iter, log=False)
                        if itr == 0: sess.graph.finalize()

                        times_outer_step.append(time.time() - time_outer_step_start)



                    ''' --------------- Logging Stuff --------------- '''

                    logger.record_tabular('Time-MAMLSteps', time.time() - time_maml_steps_start)
                    logger.record_tabular('Time-DynSampling', np.mean(times_dyn_sampling))
                    logger.record_tabular('Time-DynSampleProc', np.mean(times_dyn_sample_processing))
                    logger.record_tabular('Time-InnerStep', np.mean(times_inner_step))
                    logger.record_tabular('Time-OuterStep', np.mean(times_outer_step))


                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, all_samples_data_maml_iter[-1])  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = all_samples_data_maml_iter[-1]["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time-Overall', time.time() - start_time)
                    logger.record_tabular('Time-Itr', time.time() - itr_start_time)

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

    def initialize_uninitialized_variables(self, sess):
        uninit_vars = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        sess.run(tf.variables_initializer(uninit_vars))