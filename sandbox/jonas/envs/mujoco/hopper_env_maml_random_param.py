from rllab_maml.envs.mujoco.hopper_env import HopperEnv
from rllab_maml.core.serializable import Serializable
from sandbox.jonas.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from rllab_maml.misc import logger
from rllab_maml.misc.overrides import overrides
from sandbox.jonas.envs.helpers import get_all_function_arguments

import numpy as np


class HopperEnvMAMLRandParams(BaseEnvRandParams, HopperEnv, Serializable):

    FILE = 'hopper.xml'

    def __init__(self, *args, log_scale_limit=2.0, rand_params=BaseEnvRandParams.RAND_PARAMS, random_seed=None, **kwargs):
        """
        Ant environment with randomized mujoco parameters
        :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
        :param random_seed: random seed for sampling the mujoco model params
        :param rand_params: mujoco model parameters to sample
        """

        args_all, kwargs_all = get_all_function_arguments(self.__init__, locals())
        BaseEnvRandParams.__init__(self, *args_all, **kwargs_all)
        HopperEnv.__init__(self, *args, **kwargs)
        Serializable.__init__(self, *args_all, **kwargs_all)

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

if __name__ == "__main__":
    env = HopperEnv()
    env.reset()
    print(env.model.body_mass)
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action