from __future__ import print_function
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


class Pr2EnvLego(MujocoEnv, Serializable):

    FILE = 'pr2_legofree.xml'

    def __init__(
            self,
            model='pr2.xml',
            *args, **kwargs):

        self.goal = None
        self.action_penalty = 0.5 * 1e-2
        if model not in [None, 0]:
            self.set_model(model)

        super(Pr2EnvLego, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def set_model(self, model):
        self.__class__.FILE = model

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:-3],
            self.model.data.qvel.flat[:-3],  # Do not include the velocity of the target (should be 0).
            self.get_vec_tip_to_goal(),
        ]).reshape(-1)

    def get_tip_position(self):
        return self.model.data.site_xpos[0]

    def get_vec_tip_to_goal(self):
        tip_position = self.get_tip_position()
        goal_position = self.goal
        vec_tip_to_goal = goal_position - tip_position
        return vec_tip_to_goal

    def step(self, action):
        vec_tip_to_goal = self.get_vec_tip_to_goal()

        self.forward_dynamics(action)

        distance_tip_to_goal = np.sum(np.square(vec_tip_to_goal))

        # Penalize the robot for being far from the goal and for having the arm far from the lego.
        reward_tip = - distance_tip_to_goal
        reward_ctrl = -self.action_penalty * np.sum(np.square(action))
        # print(reward_tip)

        reward = reward_tip + reward_ctrl#+ reward_occlusion
        done = False

        ob = self.get_current_obs()

        return ob, float(reward), done, {}

    @overrides
    def reset_mujoco(self, qpos=None, qvel=None):
        qpos = self.init_qpos + np.random.normal(size=self.init_qpos.shape) * 0.01
        self.goal = np.random.uniform([0.2, -0.25, 0.5], [0.5, 0.25, 0.7])
        qpos[-3:, 0] = self.goal

        self.model.data.qpos = qpos
        self.model.data.qvel = self.init_qvel + np.random.normal(size=self.init_qvel.shape) * 0.1
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl




