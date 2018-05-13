import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param

class WalkerEnv(MujocoEnv, Serializable):

    FILE = 'walker2d.xml'

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(WalkerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
    # 0: z-com
    # 1: y-angle
    # 8: x-comvel
    # 9: z-comvel
    def get_current_obs(self):
        return np.concatenate([
            self.get_body_com("torso")[2].flat,
            self.model.data.qpos[2:].flat,
            self.get_body_comvel("torso")[[0, 2]].flat,
            self.model.data.qvel[2:].flat
        ])

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        vel = next_obs[8]
        height = next_obs[0]
        ang = next_obs[1]
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * \
            np.sum(np.square(action / scaling))
        reward = vel - ctrl_cost
        done = not (height > 0.2 and height < 1.5 and abs(ang) < 1.0)
        return Step(next_obs, reward, done)

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

    def cost_tf(self, x, u, x_next, dones):
        vel = x_next[:, 8]
        return -tf.reduce_mean((vel -
                               1e-2 * 0.5 * tf.reduce_sum(tf.square(u), axis=1)) * (1-dones)
                               )

    def cost_np_vec(self, x, u, x_next):
        vel = x_next[:, 8]
        assert np.amax(np.abs(u)) <= 1.0
        return -(vel -
                 1e-2 * 0.5 * np.sum(np.square(u), axis=1)
                 )

    def cost_np(self, x, u, x_next):
        return np.mean(self.cost_np_vec(x, u, x_next))

    def is_done(self, x, x_next):
        '''
        :param x: vector of obs
        :param x_next: vector of next obs
        :return: boolean array
        '''
        notdone = np.logical_and(
            np.logical_and(
                x_next[:, 0] >= 0.2,
                x_next[:, 0] <= 1.5
            ),
            np.abs(x_next[:, 1]) < 1.0
        )
        return np.invert(notdone)

    def is_done_tf(self, x, x_next):
        '''
        :param x:
        :param x_next:
        :return: float array 1.0 = True, 0.0 = False
        '''
        notdone = tf.logical_and(
            tf.logical_and(
                x_next[:, 0] >= 0.2,
                x_next[:, 0] <= 1.5
            ),
            tf.abs(x_next[:, 1]) < 1.0
        )
        return tf.cast(tf.logical_not(notdone), tf.float32)