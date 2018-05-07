import time

from rllab.sampler.base import BaseSampler
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger


class ModelVectorizedSampler(BaseSampler):

    def __init__(self, algo):
        super(ModelVectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs
        self.num_branches = self.algo.num_branches
        self.model_max_path_lenght = self.algo.model_max_path_length
        self.vec_env = self.algo.model_env

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, itr, real_path):
        """
        :param itr:
        :param real_path:
        :return:
        """
        # TODO: So far, I consider that I have a real world path, and that the agent never dies before the horizon
        assert len(real_path) == 1
        initial_observations = real_path[0]['observations'].reshape(-1)[:self.model_max_path_lenght] # TODO: We actually want to use the latter ones. Also, it'll be tricky in environments that it can fail before the long horizon
        initial_observations = np.repeat(initial_observations, self.num_branches)
        self.vec_env.num_envs = initial_observations.shape[0]
        logger.log("Obtaining samples for iteration %d..." % itr)
        paths = []
        obses = self.vec_env.reset(initial_observations)
        dones = np.asarray([True] * self.vec_env.num_envs)

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0
        timestep = 0

        policy = self.algo.policy
        policy.reset(dones)
        all_agent_infos = []
        all_env_infos = []
        all_observations = []
        all_rewards = []
        all_actions = []
        while timestep < self.model_max_path_lenght:
            t = time.time()
            actions, agent_infos = policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            all_agent_infos.append(agent_infos)
            all_env_infos.append(env_infos)
            all_rewards.append(rewards)
            all_actions.append(actions)
            all_observations.append(obses)

            t = time.time()
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()
        paths = [dict(observations=all_observations,
                      actions=all_actions,
                      rewards=all_rewards,
                      agent_infos=all_agent_infos,
                      env_infos=all_env_infos,
                      )]
        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)
        # TODO: If I really want the paths I need to preprocess the env_infos and agent_infos
        return paths
