from rllab.envs.base import Step
from rllab.misc.overrides import overrides
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab_maml.envs.base import Step

from rllab.core.serializable import Serializable
from sandbox.jonas.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from rllab.envs.gym_mujoco.swimmer_env import SwimmerEnv
from sandbox.jonas.envs.helpers import get_all_function_arguments



class SwimmerEnvRandParams(BaseEnvRandParams, SwimmerEnv, Serializable):

    FILE = 'swimmer.xml'

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
        SwimmerEnv.__init__(self, *args, **kwargs)
        Serializable.__init__(*args_all, **kwargs_all)

    def reward(self, obs, action, obs_next):
        ctrl_cost_coeff = 0.0001
        if obs.ndim == 2 and action.ndim == 2:
            vel = obs_next[:, 3]
            ctrl_cost = ctrl_cost_coeff * np.sum(np.square(action), axis=1)
            reward = vel - ctrl_cost
        else:
            reward = self.reward(np.array([obs]), np.array([action]), np.array([obs_next]))[0]
        return np.minimum(np.maximum(-1000.0, reward), 1000.0)


    @overrides
    def log_diagnostics(self, paths, prefix=''):
        if len(paths) > 0:
            progs = [
                np.linalg.norm(path["observations"][:, -3:], axis=1)
                for path in paths
            ]
            logger.record_tabular(prefix +'AverageForwardProgress', np.mean(progs))
            logger.record_tabular(prefix + 'MaxForwardProgress', np.max(progs))
            logger.record_tabular(prefix + 'MinForwardProgress', np.min(progs))
            logger.record_tabular(prefix + 'StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular(prefix + 'AverageForwardProgress', np.nan)
            logger.record_tabular(prefix + 'MaxForwardProgress', np.nan)
            logger.record_tabular(prefix + 'MinForwardProgress', np.nan)
            logger.record_tabular(prefix + 'StdForwardProgress', np.nan)
