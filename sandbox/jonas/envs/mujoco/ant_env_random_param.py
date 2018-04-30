from rllab.envs.mujoco.ant_env import AntEnv
from rllab.core.serializable import Serializable
from sandbox.jonas.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from sandbox.jonas.envs.helpers import get_all_function_arguments

import numpy as np



class AntEnvRandParams(BaseEnvRandParams, AntEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def __init__(self, *args, log_scale_limit=2.0, fix_params=False, rand_params=BaseEnvRandParams.RAND_PARAMS, random_seed=None, **kwargs):
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

if __name__ == "__main__":
    env = AntEnvRandParams()
    env.reset()
    print(env.model.body_mass)
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action