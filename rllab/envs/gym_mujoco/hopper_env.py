import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.gym_mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides


# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)


class HopperEnv(MujocoEnv, Serializable):

    FILE = 'hopper.xml'

    @autoargs.arg('alive_coeff', type=float,
                  help='reward coefficient for being alive')
    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            alive_coeff=1,
            ctrl_cost_coeff=0.01,
            *args, **kwargs):
        self.alive_coeff = alive_coeff
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(HopperEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    @overrides
    def step(self, action):
        posbefore = self.model.data.qpos[0]
        self.forward_dynamics(action)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()
        ob = self.get_current_obs()
        done = not (np.isfinite(ob).all() and (np.abs(ob[1:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))

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
