from rllab_maml.envs.mujoco.gather.gather_env import GatherEnv
from rllab_maml.envs.mujoco.swimmer_env import SwimmerEnv


class SwimmerGatherEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2
