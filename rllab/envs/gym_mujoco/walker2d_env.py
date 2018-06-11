import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.gym_mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class Walker2DEnv(MujocoEnv, Serializable):

    FILE = 'walker2d.xml'

    def __init__(
            self, *args, **kwargs):
        super(Walker2DEnv, self).__init__(*args, **kwargs)
        self.frame_skip = 4
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[1:].flat,
            np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    def step(self, action):
        posbefore = self.model.data.qpos[0]
        self.forward_dynamics(action)
        posafter, height, ang = self.model.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self.get_current_obs()

        self.time_step += 1
        if self.max_path_length and self.time_step >= self.max_path_length:
            done = True

        return ob, float(reward), done, {}

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

