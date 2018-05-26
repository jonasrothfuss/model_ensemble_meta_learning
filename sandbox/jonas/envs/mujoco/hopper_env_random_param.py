from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.core.serializable import Serializable
from sandbox.jonas.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from sandbox.jonas.envs.helpers import get_all_function_arguments
from rllab.misc.overrides import overrides
from rllab_maml.envs.base import Step
from rllab.misc import logger

import numpy as np



class HopperEnvRandParams(BaseEnvRandParams, HopperEnv, Serializable):

    FILE = 'hopper.xml'

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
        HopperEnv.__init__(self, *args, **kwargs)
        Serializable.__init__(*args_all, **kwargs_all)

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        vel = self.get_body_comvel("torso")[0]
        reward = vel + self.alive_coeff - \
            0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        state = self._state
        notdone = np.isfinite(state).all() and \
            (np.abs(state[3:]) < 100).all() and (state[0] > .7) and \
            (abs(state[2]) < .2)
        self.n_steps += 1
        done = not notdone or self.n_steps >= self.max_path_length
        # clip reward in case mujoco sim goes crazy
        reward = np.minimum(np.maximum(-1000.0, reward), 1000.0)

        return Step(next_obs, reward, done)

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

    def reward(self, obs, action, obs_next):
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        if obs.ndim == 2 and action.ndim == 2:
            assert obs.shape == obs_next.shape and action.shape[0] == obs.shape[0]
            vel = (obs_next[:, -3] - obs[:, -3]) / 0.02
            ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling), axis=1)
            vel + self.alive_coeff - ctrl_cost
            return vel + self.alive_coeff - ctrl_cost
        else:
            vel = (obs_next[-3] - obs[-3])/0.02
            ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
            return vel + self.alive_coeff - ctrl_cost

    def done(self, obs):
        if obs.ndim == 2:
            notdone = np.all(np.isfinite(obs), axis=1) * (np.abs(obs[:, 3:]) < 100).all(axis=1) * (obs[:, 0] > .7) * (np.abs(obs[:, 1]) < .2)
            return np.logical_not(notdone)
        else:
            notdone = np.isfinite(obs).all() and \
                      (np.abs(obs[3:]) < 100).all() and (obs[0] > .7) and \
                      (abs(obs[1]) < .2)
            return not notdone

if __name__ == "__main__":

    env = HopperEnvRandParams()
    env.reset()
    print(env.model.body_mass)
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
