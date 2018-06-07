import numpy as np
from gym import utils
from rllab.envs.gym_mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

class ReacherEnv(MujocoEnv, Serializable):

    FILE = 'hopper.xml'

    def __init__(self, *args, **kwargs):
        super(ReacherEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    @overrides
    def get_current_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta), #joints
            np.sin(theta), #joints
            self.model.data.qpos.flat[2:], #target position
            self.model.data.qvel.flat[:2], # joint volocities
            self.get_body_com("fingertip") - self.get_body_com("target") #distance to target
        ])

    def step(self, action):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + reward_ctrl
        self.forward_dynamics(action)
        ob = self.get_current_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    @overrides
    def reset_mujoco(self, init_state=None):
        if init_state is None:
            qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flatten()
            while True:
                self.goal = np.random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(self.goal) < 2:
                    break
            qpos[-2:] = self.goal
            qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.model.nv)
            qvel[-2:] = 0
            self.model.data.qpos = qpos
            self.model.data.qvel = qvel
            return self.get_current_obs()
        else:
            raise NotImplementedError

