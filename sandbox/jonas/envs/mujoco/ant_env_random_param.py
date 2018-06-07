from rllab.envs.mujoco.ant_env import AntEnv
from rllab.core.serializable import Serializable
from sandbox.jonas.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from sandbox.jonas.envs.helpers import get_all_function_arguments

from rllab.misc.overrides import overrides
from rllab.envs.base import Step
from rllab.misc import logger

import numpy as np



class AntEnvRandParams(BaseEnvRandParams, AntEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

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
        AntEnv.__init__(self, *args, **kwargs)
        Serializable.__init__(*args_all, **kwargs_all)


    def reward(self, obs, action, obs_next):
        if obs.ndim == 2 and action.ndim == 2:
            assert obs.shape == obs_next.shape and action.shape[0] == obs.shape[0]
            forward_vel = obs_next[:, 13]
            lb, ub = self.action_bounds
            scaling = (ub - lb) * 0.5
            ctrl_cost = .5 * 1e-2 * np.square(action/scaling).sum()
            survive_reward = 1.0
            return forward_vel - ctrl_cost + survive_reward
        else:
            return self.reward(np.array([obs]), np.array([action]), np.array([obs_next]))[0]

    def done(self, obs):
        if obs.ndim == 2:
            notdone = np.all(np.isfinite(obs), axis=1) * (obs[:, 0] >= 0.2) * (obs[:, 0] <= 1.0)
            return np.logical_not(notdone)
        else:
            notdone = np.isfinite(obs).all()  and obs[0] >= 0.2 and obs[0] <= 1.0
            return not notdone


    def _obs_bounds(self):
        self._obs_lower_bounds = -1000 * np.ones(shape=(self.model.data.qpos.shape[0] + self.model.data.qvel.shape[0]-2,))
        self._obs_upper_bounds = 1000 * np.ones(shape=(self.model.data.qpos.shape[0] + self.model.data.qvel.shape[0]-1,))

    @property
    def obs_lower_bounds(self):
        return self._obs_lower_bounds

    @property
    def obs_upper_bounds(self):
        return self._obs_upper_bounds

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
    env = AntEnvRandParams()
    env.reset()
    print(env.model.body_mass)
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action