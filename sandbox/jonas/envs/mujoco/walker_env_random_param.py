from rllab.envs.base import Step
from rllab.misc.overrides import overrides
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab_maml.envs.base import Step

from rllab.core.serializable import Serializable
from sandbox.jonas.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from rllab.envs.gym_mujoco.walker2d_env import Walker2DEnv
from sandbox.jonas.envs.helpers import get_all_function_arguments



class WalkerEnvRandomParams(BaseEnvRandParams, Walker2DEnv, Serializable):

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
        Walker2DEnv.__init__(self, *args, **kwargs)
        Serializable.__init__(*args_all, **kwargs_all)
        self._obs_bounds()

    def reward(self, obs, action, obs_next):
        if obs.ndim == 2 and action.ndim == 2:
            vel = obs_next[:, 8]
            alive_bonus = 1.0
            ctrl_cost = 1e-3 * np.sum(np.square(action), axis=1)
            reward = vel - ctrl_cost + alive_bonus
        else:
            reward = self.reward(np.array([obs]), np.array([action]), np.array([obs_next]))[0]
        return reward

    def done(self, obs):
        if obs.ndim == 2:
            notdone = (obs[:, 0] > 0.8) *  (obs[:, 0] < 2.0) * (obs[:, 1] > -1.0) * (obs[:, 1] < 1.0)
            return np.logical_not(notdone)
        else:
            return not (obs[0] > 0.8 and obs[0] < 2.0 and obs[1] > -1.0 and obs[1] < 1.0)

    def _obs_bounds(self):
        jnt_range = self.model.jnt_range
        jnt_limited = self.model.jnt_limited
        self._obs_lower_bounds = -10 * np.ones(
            shape=(self.model.data.qpos.shape[0] + self.model.data.qvel.shape[0] - 1,))
        self._obs_upper_bounds = 10 * np.ones(
            shape=(self.model.data.qpos.shape[0] + self.model.data.qvel.shape[0] - 1,))
        for idx, limited in enumerate(jnt_limited):
            if idx > 0 and limited:
                self._obs_lower_bounds[idx] = jnt_range[idx][0]
                self._obs_upper_bounds[idx] = jnt_range[idx][1]

    @property
    def obs_lower_bounds(self):
        return self._obs_lower_bounds

    @property
    def obs_upper_bounds(self):
        return self._obs_upper_bounds

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        if len(paths) > 0:
            progs = [
                path["observations"][3]
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
