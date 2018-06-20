from __future__ import print_function
from rllab.envs.gym_mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


class PR2Env(MujocoEnv, Serializable):

    FILE = 'pr2_legofree.xml'

    def __init__(
            self,
            model='pr2.xml',
            *args, **kwargs):

        self.goal = None
        self.action_penalty = 0.5 * 1e-2
        if model not in [None, 0]:
            self.set_model(model)

        super(PR2Env, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    from __future__ import print_function
    from rllab.envs.gym_mujoco.mujoco_env import MujocoEnv
    import numpy as np
    from rllab.core.serializable import Serializable
    from rllab.misc.overrides import overrides
    from rllab.mujoco_py import MjModel, MjViewer

    class PR2Env(MujocoEnv, Serializable):

        FILE = 'pr2_legofree.xml'

        def __init__(
                self,
                model='pr2.xml',
                *args, **kwargs):

            self.goal = None
            self.action_penalty = 0.5 * 1e-2
            if model not in [None, 0]:
                self.set_model(model)

            super(PR2Env, self).__init__(*args, **kwargs)
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
            lb, ub = self.action_bounds
            scaling = (ub - lb) * 0.5
            reward_ctrl = -self.action_penalty * np.sum(np.square(action / scaling))

            distance_tip_to_goal = np.sum(np.square(vec_tip_to_goal))
            reward_tip = - distance_tip_to_goal

            reward = reward_tip + reward_ctrl  # + reward_occlusion
            done = False

            self.time_step += 1
            if self.max_path_length and self.time_step > self.max_path_length:
                done = True

            ob = self.get_current_obs()

            return ob, float(reward), done, {}

        def get_viewer(self):
            if self.viewer is None:
                self.viewer = MjViewer()
                self.viewer.start()
                self.viewer.set_model(self.model)
                self.viewer.cam.camid = 0
            return self.viewer

        @overrides
        def reset_mujoco(self, qpos=None, qvel=None):
            qpos = self.init_qpos + np.random.normal(size=self.init_qpos.shape) * 0.01
            self.goal = np.random.uniform([0.4, 0.25, 0.6], [0.6, 0.75, 1.])
            qpos[-3:, 0] = self.goal

            qvel = self.init_qvel + np.random.normal(size=self.init_qvel.shape) * 0.1
            qvel[-3:, 0] = 0

            self.model.data.qpos = qpos
            self.model.data.qvel = qvel
            self.model.data.qacc = self.init_qacc

    self.model.data.ctrl = self.init_ctrl
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
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        reward_ctrl = -self.action_penalty * np.sum(np.square(action/scaling))

        distance_tip_to_goal = np.sum(np.square(vec_tip_to_goal))
        reward_tip = - distance_tip_to_goal

        reward = reward_tip + reward_ctrl#+ reward_occlusion
        done = False

        self.time_step += 1
        if self.max_path_length and self.time_step > self.max_path_length:
            done = True

        ob = self.get_current_obs()

        return ob, float(reward), done, {}

    @overrides
    def reset_mujoco(self, qpos=None, qvel=None):
        qpos = self.init_qpos + np.random.normal(size=self.init_qpos.shape) * 0.01
        self.goal = np.random.uniform([0.4, 0.25, 0.6], [0.6, 0.75, 1.])
        qpos[-3:, 0] = self.goal

        qvel = self.init_qvel + np.random.normal(size=self.init_qvel.shape) * 0.1
        qvel[-3:, 0] = 0

        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl




