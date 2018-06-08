import pickle

from sandbox.jonas.sampler.base import ModelBaseSampler
from sandbox.jonas.sampler.MAML_model_sampler.maml_model_vec_env_executor import MAMLModelVecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools

class MAMLModelVectorizedSampler(ModelBaseSampler):

    def __init__(self, algo, n_parallel=None, max_path_length=None, clip_obs=False):
        """
        :param algo: RL algo
        :param n_parallel: number of trajectories samples in parallel
        """
        super(MAMLModelVectorizedSampler, self).__init__(algo)
        self.n_models = self.algo.dynamics_model.num_models
        self.meta_batch_size = self.algo.meta_batch_size
        self.clip_obs = clip_obs

        if n_parallel is None:
            self.n_parallel = self.algo.batch_size_dynamics_samples // self.algo.max_path_length
        else:
            self.n_parallel = n_parallel

        if max_path_length is None:
            self.max_path_length = self.algo.max_path_length
        else:
            self.max_path_length = max_path_length

        assert self.n_parallel % self.n_models == 0

    def start_worker(self):
        env = pickle.loads(pickle.dumps(self.algo.env))

        self.vec_env = MAMLModelVecEnvExecutor(
            env=env,
            model=self.algo.dynamics_model,
            max_path_length=self.max_path_length,
            n_parallel=self.n_parallel,
            clip_obs=self.clip_obs,
        )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr, return_dict=False, log=True, log_prefix='', traj_starting_obs=None,
                       traj_starting_ts=None):
        """

        :param itr: current iteration (int) for logging purposes
        :param return_dict: (boolean) weather to return a dict or a list
        :param log: (boolean) indicates whether to log
        :param log_prefix: (str) prefix to prepend to the log keys
        :param traj_starting_obs: (optional) starting observations to randomly choose from for rolling out trajectories [numpy array of shape (n_observations, ndim_obs),
                                    if env.reset() is called to get a initial observations
        :return:
        """
        # return_dict: whether or not to return a dictionary or list form of paths
        assert traj_starting_obs is None or traj_starting_obs.ndim == 2

        paths = {}
        for i in range(self.meta_batch_size):
            paths[i] = []

        n_samples = 0
        n_parallel_per_task = self.vec_env.num_envs // self.meta_batch_size

        obses = self.vec_env.reset(traj_starting_obs=traj_starting_obs)
        dones = np.asarray([True] * self.n_parallel)
        running_paths = [None] * self.n_parallel

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        import time

        while n_samples < self.algo.batch_size_dynamics_samples:
            t = time.time()
            policy.reset(dones)

            # get actions from MAML policy
            obs_per_task = np.split(np.asarray(obses), self.meta_batch_size)
            actions, agent_infos = policy.get_actions_batch(obs_per_task)

            assert actions.shape[0] == self.n_parallel

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions, traj_starting_obs=traj_starting_obs,
                                                                      traj_starting_ts=traj_starting_ts)
            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths[idx // n_parallel_per_task].append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        if log:
            logger.record_tabular(log_prefix + "PolicyExecTime", policy_time)
            logger.record_tabular(log_prefix + "EnvExecTime", env_time)
            logger.record_tabular(log_prefix + "ProcessExecTime", process_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())
            # path_keys = flatten_list([[key]*len(paths[key]) for key in paths.keys()])
        else:
            assert len(paths) == self.meta_batch_size
        return paths
