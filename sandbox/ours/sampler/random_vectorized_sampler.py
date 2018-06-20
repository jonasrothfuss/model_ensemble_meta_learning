import pickle

from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab_maml.misc.overrides import overrides
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools

from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from sandbox.ours.sampler.base import RandomBaseSampler

class RandomVectorizedSampler(RandomBaseSampler, VectorizedSampler):

    def __init__(self, algo, n_envs=None):
        self.algo = algo
        VectorizedSampler.__init__(self, algo, n_envs=n_envs)

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length
            )
        self.env_spec = self.algo.env.spec


    def shutdown_worker(self):
        self.vec_env.terminate()

    @overrides
    def obtain_samples(self, itr, num_samples=None, log=True, log_prefix='RandomSampler-'):
        if num_samples is None:
            num_samples = self.algo.batch_size

        paths = []
        n_samples_collected = 0
        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(num_samples)
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        import time
        while n_samples_collected < num_samples:
            # random actions
            t = time.time()
            actions = np.stack([self.vec_env.action_space.sample() for _ in range(len(obses))], axis=0)
            policy_time = time.time() - t
            agent_infos = {}

            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
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
                    paths.append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples_collected += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        if log:
            logger.record_tabular(log_prefix + "PolicyExecTime", policy_time)
            logger.record_tabular(log_prefix + "EnvExecTime", env_time)
            logger.record_tabular(log_prefix + "ProcessExecTime", process_time)

        return paths