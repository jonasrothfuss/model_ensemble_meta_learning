import numpy as np
import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahEnv(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, **kwargs):
        super(HalfCheetahEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.ctrl_cost_coeff = 1e-1

    def get_current_obs(self):
        return np.concatenate([
            self.get_body_com("torso")[[0,2]],
            self.model.data.qpos.flatten()[2:],
            self.get_body_comvel("torso")[[0,2]],
            self.model.data.qvel.flatten()[2:],
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = self.ctrl_cost_coeff * 0.5 * np.sum(np.square(action))
        run_cost = -1 * self.get_body_comvel("torso")[0]
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return Step(next_obs, reward, done, real=True)

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

    def is_done(self, obs):
        return [False] * len(obs)

    def reward_np(self, x, u, x_next):
        return x_next[:, 9] - self.ctrl_cost_coeff * 0.5 * np.sum(np.square(u), axis=1)

    def cost_tf(self, x, u, x_next):
        return tf.clip_by_value(-tf.reduce_mean(x_next[:, 9] - self.ctrl_cost_coeff * 0.5 * tf.reduce_sum(tf.square(u), axis=1)), -10, 10)

    def cost_np_vec(self, x, u, x_next):
        assert np.amax(np.abs(u)) <= 1.0 + 1e-6
        return np.clip(-(x_next[:, 9] - self.ctrl_cost_coeff * 0.5 * np.sum(np.square(u), axis=1)), -10, 10)