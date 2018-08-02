import pickle

import tensorflow as tf
from sandbox.ours.sampler.base import MAMLBaseSampler
from sandbox_maml.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.dennis.sampler.maml_parallel_env_executor import MAMLParallelVecEnvExecutor
from sandbox.ours.sampler.MAML_sampler.maml_vec_env_executor import MAMLVecEnvExecutor
from rllab_maml.misc import tensor_utils
import numpy as np
from rllab_maml.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools


class MAMLVectorizedSampler(MAMLBaseSampler):

    def __init__(self, algo, n_tasks, n_envs=None, parallel=True):
        super(MAMLVectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs
        self.n_tasks = n_tasks
        self.parallel = parallel
        self.env_spec = self.algo.env.spec
        self.vec_env = None

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        elif self.parallel:
            self.vec_env = MAMLParallelVecEnvExecutor(self.algo.env, self.n_tasks, self.n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(self.n_envs)]
            self.vec_env = MAMLVecEnvExecutor(
                envs=envs,
                #env=pickle.loads(pickle.dumps(self.algo.env)),
                #n = n_envs,
                max_path_length=self.algo.max_path_length
            )

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr, reset_args=None, return_dict=False, log_prefix=''):
        # reset_args: arguments to pass to the environments to reset
        # return_dict: whether or not to return a dictionary or list form of paths

        logger.log("Obtaining samples for iteration %d..." % itr)

        #paths = []
        paths = {}
        for i in range(self.n_tasks):
            paths[i] = []

        assert self.vec_env.num_envs % self.n_tasks == 0

        n_envs_per_task = self.vec_env.num_envs // self.n_tasks

        # if the reset args are not list/numpy, we set the same args for each env
        if reset_args is not None and (type(reset_args) != list and type(reset_args)!=np.ndarray):
            reset_args = [reset_args]*self.vec_env.num_envs
        elif reset_args is not None:
            # duplicate reset_args as n_envs_per_task times for each task
            assert len(reset_args) == self.n_tasks
            reset_args = [reset_arg for reset_arg in reset_args for _ in range(n_envs_per_task)]
            assert len(reset_args) == self.vec_env.num_envs
        else:
            raise AssertionError("reset args must not be none")

        n_samples = 0
        obses = self.vec_env.reset(reset_args)
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        import time

        while n_samples < self.algo.batch_size:
            t = time.time()
            policy.reset(dones)

            obs_per_task = np.split(np.asarray(obses), self.n_tasks)
            actions, agent_infos = policy.get_actions_batch(obs_per_task)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions, reset_args)
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
                    paths[idx // n_envs_per_task].append(dict(
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

        logger.record_tabular(log_prefix+"PolicyExecTime", policy_time)
        logger.record_tabular(log_prefix+"EnvExecTime", env_time)
        logger.record_tabular(log_prefix+"ProcessExecTime", process_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())
            #path_keys = flatten_list([[key]*len(paths[key]) for key in paths.keys()])
        return paths