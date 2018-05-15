from rllab_maml.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab_maml.core.serializable import Serializable
from sandbox.jonas.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from rllab_maml.misc import logger
from rllab_maml.misc.overrides import overrides
from sandbox.jonas.envs.helpers import get_all_function_arguments
from rllab_maml.envs.base import Step

import numpy as np


class HalfCheetahMAMLEnvRandParams(BaseEnvRandParams, HalfCheetahEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, log_scale_limit=2.0, fix_params=False, rand_params=BaseEnvRandParams.RAND_PARAMS, random_seed=None, **kwargs):
        """
        Half-Cheetah environment with randomized mujoco parameters
        :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
        :param random_seed: random seed for sampling the mujoco model params
        :param fix_params: boolean indicating whether the mujoco parameters shall be fixed
        :param rand_params: mujoco model parameters to sample
        """
        self.ctrl_cost_coeff = 1e-1
        args_all, kwargs_all = get_all_function_arguments(self.__init__, locals())
        BaseEnvRandParams.__init__(*args_all, **kwargs_all)
        HalfCheetahEnv.__init__(self, *args, **kwargs)
        Serializable.__init__(*args_all, **kwargs_all)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-4] - path["observations"][0][-4]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = self.ctrl_cost_coeff * 0.5 * np.sum(np.square(action))
        run_cost = -1 * self.get_body_comvel("torso")[0]
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False

        # clip reward in case mujoco sim goes crazy
        reward = np.minimum(np.maximum(-1000.0, reward), 1000.0)

        return Step(next_obs, reward, done)

    def reward(self, obs, action, obs_next):
        if obs.ndim == 2 and action.ndim == 2:
            assert obs.shape == obs_next.shape and action.shape[0] == obs.shape[0]
            forward_vel = (obs_next[:, -3] - obs[:, -3]) / 0.01
            ctrl_cost = self.ctrl_cost_coeff * 0.5 * np.sum(np.square(action), axis=1)
            return forward_vel - ctrl_cost
        else:
            forward_vel = (obs_next[-3] - obs[-3]) / 0.01
            ctrl_cost = self.ctrl_cost_coeff * 0.5 * np.sum(np.square(action))
            return forward_vel - ctrl_cost

if __name__ == "__main__":
    env = HalfCheetahMAMLEnvRandParams()
    env.reset()
    print(env.model.body_mass)
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action