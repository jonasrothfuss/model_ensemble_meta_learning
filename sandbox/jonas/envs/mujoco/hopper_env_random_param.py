from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.core.serializable import Serializable
from sandbox.jonas.envs.mujoco.base_env_rand_param import BaseEnvRandParams

import numpy as np



class HopperEnvRandParams(BaseEnvRandParams, HopperEnv, Serializable):

    FILE = 'hopper.xml'

    def __init__(self, *args, log_scale_limit=2.0, random_seed=None, **kwargs):
        """
        Half-Cheetah environment with randomized mujoco parameters
        :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
        :param random_seed: random seed for sampling the mujoco model params
        """
        self.log_scale_limit = log_scale_limit
        self.random_state = np.random.RandomState(random_seed)
        self.fixed_params = False # can be changed by calling the fix_mujoco_parameters method

        super(HopperEnvRandParams, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)



if __name__ == "__main__":

    env = HopperEnvRandParams()
    env.reset()
    print(env.model.body_mass)
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
