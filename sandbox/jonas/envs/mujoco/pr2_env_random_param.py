from rllab.envs.gym_mujoco.pr2_env import PR2Env
from rllab.core.serializable import Serializable
from sandbox.jonas.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from sandbox.jonas.envs.helpers import get_all_function_arguments

from rllab.misc.overrides import overrides
from rllab.envs.base import Step
from rllab.misc import logger

import numpy as np



class PR2EnvRandParams(BaseEnvRandParams, PR2Env, Serializable):

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
        PR2Env.__init__(self, *args, **kwargs)
        Serializable.__init__(*args_all, **kwargs_all)


    def reward(self, obs, action, obs_next):
        if obs.ndim == 2 and action.ndim == 2:
            assert obs.shape == obs_next.shape and action.shape[0] == obs.shape[0]
            distance_tip_to_goal = np.sum(np.square(obs_next[:, -3:]), axis=1)
            lb, ub = self.action_bounds
            scaling = (ub - lb) * 0.5
            ctrl_cost = .5 * 1e-2 * np.sum(np.square(action/scaling), axis=1)
            return -distance_tip_to_goal - ctrl_cost
        else:
            return self.reward(np.array([obs]), np.array([action]), np.array([obs_next]))[0]

    def _obs_bounds(self):
        self._obs_lower_bounds = -1000 * np.ones(shape=(self.model.data.qpos.shape[0] + self.model.data.qvel.shape[0]-2,))
        self._obs_upper_bounds = 1000 * np.ones(shape=(self.model.data.qpos.shape[0] + self.model.data.qvel.shape[0]-1,))

    @property
    def obs_lower_bounds(self):
        return self._obs_lower_bounds

    @property
    def obs_upper_bounds(self):
        return self._obs_upper_bounds

if __name__ == "__main__":
    env = PR2EnvRandParams()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            env.step(env.action_space.sample())  # take a random action