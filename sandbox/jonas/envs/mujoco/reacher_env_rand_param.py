from rllab.envs.gym_mujoco.reacher_env import ReacherEnv
from rllab.core.serializable import Serializable
from sandbox.jonas.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from sandbox.jonas.envs.helpers import get_all_function_arguments
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab_maml.envs.base import Step
from rllab.misc import logger

import numpy as np

class ReacherEnvRandParams(BaseEnvRandParams, ReacherEnv, Serializable):

    FILE = 'reacher.xml'

    def __init__(self, *args, log_scale_limit=2.0, fix_params=False, rand_params=BaseEnvRandParams.RAND_PARAMS, random_seed=None, max_path_length=None, **kwargs):
        """
        Half-Cheetah environment with randomized mujoco parameters
        :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
        :param random_seed: random seed for sampling the mujoco model params
        :param fix_params: boolean indicating whether the mujoco parameters shall be fixed
        :param rand_params: mujoco model parameters to sample
        """
        args_all, kwargs_all = get_all_function_arguments(self.__init__, locals())
        BaseEnvRandParams.__init__(*args_all, **kwargs_all)
        ReacherEnv.__init__(self, *args, **kwargs)
        Serializable.__init__(*args_all, **kwargs_all)

    def reward(self, obs, action, obs_next):
        if obs.ndim == 2 and action.ndim == 2:
            assert obs.shape == obs_next.shape and action.shape[0] == obs.shape[0]
            reward_dist = - np.linalg.norm(obs[:, -3:], axis=1)
            reward_ctrl = - np.sum(np.square(action), axis=1)
            reward = reward_dist + reward_ctrl
            return np.minimum(np.maximum(-1000.0, reward), 1000.0)
        else:
            reward_dist = - np.linalg.norm(obs[-3:])
            reward_ctrl = - np.square(action).sum()
            reward = reward_dist + reward_ctrl
            return np.minimum(np.maximum(-1000.0, reward), 1000.0)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))

if __name__ == "__main__":
    env = ReacherEnvRandParams()
    env.reset()
    print(env.model.body_mass)
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())  # take a random action
