from rllab.envs.gym_mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math


class AntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def __init__(self, *args, **kwargs):
        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.frame_skip = 2

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        forward_reward = self.model.data.qvel[0, 0] # velocity in x direction
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * 1e-2 * np.square(action/scaling).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost + survive_reward
        ob = self.get_current_obs()
        notdone = np.isfinite(ob).all() and ob[0] <= 1.0 and ob[0] >= 0.2
        done = not notdone

        reward = np.minimum(np.maximum(-1000.0, reward), 1000.0)

        self.time_step += 1
        if self.max_path_length and self.time_step > self.max_path_length:
            done = True

        return ob, reward, done, {}

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

