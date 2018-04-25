from rllab.envs.mujoco.ant_env import AntEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
from base_env_rand_param import BaseEnvRandParams

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math
import warnings


class AntEnvRandParams(BaseEnvRandParams, AntEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def __init__(self, log_scale_limit=2.0, random_seed=None, *args, **kwargs):
        """
        Ant environment with randomized mujoco parameters
        :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
        :param random_seed: random seed for sampling the mujoco model params
        """
        self.log_scale_limit = log_scale_limit
        self.random_state = np.random.RandomState(random_seed)
        self.fixed_params = False # can be changed by calling the fix_mujoco_parameters method

        super(AntEnvRandParams, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
