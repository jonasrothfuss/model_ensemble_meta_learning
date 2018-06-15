from rllab.envs.base import Step
from rllab.misc.overrides import overrides
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab_maml.envs.base import Step

from rllab.core.serializable import Serializable
from sandbox.ours.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from rllab.envs.mujoco.snake_env import SnakeEnv
from sandbox.ours.envs.helpers import get_all_function_arguments

idx = 7 #idx position of velocity in obs

class SnakeEnvRandParams(BaseEnvRandParams, SnakeEnv, Serializable):

    FILE = 'snake.xml'

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
        SnakeEnv.__init__(self, *args, **kwargs)
        Serializable.__init__(*args_all, **kwargs_all)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        done = False
        return Step(next_obs, reward, done)

    def reward(self, obs, action, obs_next):
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        if obs.ndim == 2 and action.ndim == 2:
            ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling), axis=1)
            vel = obs_next[:, 7]
            reward = vel - ctrl_cost
        else:
            ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
            vel = obs_next[7]
            reward = vel - ctrl_cost
        return np.minimum(np.maximum(-1000.0, reward), 1000.0)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular('AverageForwardProgress', np.nan)
            logger.record_tabular('MaxForwardProgress', np.nan)
            logger.record_tabular('MinForwardProgress', np.nan)
            logger.record_tabular('StdForwardProgress', np.nan)

if __name__ == "__main__":

    env = SnakeEnvRandParams()
    env.reset()
    print(env.model.body_mass)
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
