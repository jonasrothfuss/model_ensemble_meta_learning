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
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
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

    def reset_mujoco(self, init_obs=None):
        if init_obs is None:
            super(HalfCheetahEnv, self).reset_mujoco()
        else:
            start = 0
            init_obs = np.concatenate([[0], init_obs])
            for datum_name in ["qpos", "qvel"]:
                datum = getattr(self.model.data, datum_name)
                datum_dim = datum.shape[0]
                datum = init_obs[start: start + datum_dim]
                setattr(self.model.data, datum_name, datum)
                start += datum_dim



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

    def reward_np(self, obs, action, obs_next):
        if obs.ndim == 2 and action.ndim == 2:
            assert obs.shape == obs_next.shape and action.shape[0] == obs.shape[0]
            forward_vel = (obs_next[:, -3] - obs[:, -3]) / 0.01
            ctrl_cost = self.ctrl_cost_coeff * 0.5 * np.sum(np.square(action), axis=1)
            return forward_vel - ctrl_cost
        else:
            forward_vel = (obs_next[-3] - obs[-3]) / 0.01
            ctrl_cost = self.ctrl_cost_coeff * 0.5 * np.sum(np.square(action))
            return forward_vel - ctrl_cost

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
