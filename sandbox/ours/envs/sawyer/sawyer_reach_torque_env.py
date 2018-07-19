from collections import OrderedDict
import numpy as np
from sandbox.ours.envs.sawyer.mujoco_env import MujocoEnv
from gym.spaces import Dict
from rllab.spaces import Box
from rllab.core.serializable import Serializable
from rllab.misc import logger

from sandbox.ours.envs.sawyer.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from sandbox.ours.envs.sawyer.core.multitask_env import MultitaskEnv


class SawyerReachTorqueEnv(MujocoEnv, Serializable, MultitaskEnv):
    """Implements a torque-controlled Sawyer environment"""

    def __init__(self,
                 frame_skip=10,
                 action_scale=10,
                 xyz_obs=False, #TODO: maybe delete this observation since xyz_obs only doesn't work
                 keep_vel_in_obs=True,
                 use_safety_box=False,
                 fix_goal=False,
                 fixed_goal=(0.05, 0.6, 0.15),
                 reward_type='hand_distance',
                 ctrl_cost_coef=0.0,
                 indicator_threshold=.05,
                 goal_low=None,
                 goal_high=None,
                 ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        self.action_scale = action_scale
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = Box(low=low, high=high)
        if goal_low is None:
            goal_low = np.array([-0.1, 0.5, 0.02])
        else:
            goal_low = np.array(goal_low)
        if goal_high is None:
            goal_high = np.array([0.1, 0.7, 0.2])
        else:
            goal_high = np.array(goal_low)
        self.safety_box = Box(
            goal_low,
            goal_high
        )
        self.xyz_obs = xyz_obs
        self.keep_vel_in_obs = keep_vel_in_obs
        self.ctrl_cost_coef = ctrl_cost_coef
        self.goal_space = Box(goal_low, goal_high)
        obs_size = self._get_env_obs().shape[0]
        high = np.inf * np.ones(obs_size)
        low = -high
        self.obs_space = Box(low, high)
        self.achieved_goal_space = Box(
            -np.inf * np.ones(3),
            np.inf * np.ones(3)
        )
        self._observation_space_dict = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.achieved_goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.achieved_goal_space),
        ])

        # obs space: (current_xyz_pos, desired_xyz_pos)
        self.observation_space = Box(np.concatenate([high, goal_low]),
                                     np.concatenate([low, goal_high]))
        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self.use_safety_box=use_safety_box
        self.prev_qpos = self.init_angles.copy()
        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        self.reset()

    @property
    def model_name(self):
       return 'sawyer_reach_torque.xml'

    def reset_to_prev_qpos(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.prev_qpos.copy()
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        self.set_goal_xyz(self._state_goal)

    def is_outside_box(self):
        pos = self.get_endeff_pos()
        return not self.safety_box.contains(pos)

    def set_to_qpos(self, qpos):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = qpos
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        self.set_goal_xyz(self._state_goal)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

        # 3rd person view
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        action = action * self.action_scale
        self.do_simulation(action, self.frame_skip)
        if self.use_safety_box:
            if self.is_outside_box():
                self.reset_to_prev_qpos()
            else:
                self.prev_qpos = self.data.qpos.copy()
        obs_dict = self._get_obs_dict()
        info = self._get_info()
        reward = self.compute_reward(action, obs_dict)
        obs = self._convert_obs_dict_to_obs(obs_dict)
        done = False
        return obs, reward, done, info

    def reward(self, obs, action, obs_next):
        if obs_next.ndim == 2 and action.ndim == 2:
            hand_pos = obs_next[:, 0:3]
            goals = obs_next[:, -3:]
            distance = np.linalg.norm(hand_pos - goals, axis=1)
            ctrl_cost = self.ctrl_cost_coef * np.sum(np.abs(action), axis=1)
            if self.reward_type == 'hand_distance':
                r = -distance
            elif self.reward_type == 'hand_success':
                r = -(distance < self.indicator_threshold).astype(float)
            else:
                raise NotImplementedError("Invalid/no reward type.")
            return r - ctrl_cost
        else:
            return self.reward(np.array([obs]), np.array([action]), np.array([obs_next]))[0]

    def _get_env_obs(self):
        if self.xyz_obs:
            if self.keep_vel_in_obs:
                return np.concatenate([
                    self.get_endeff_pos(), # end-effector xyz position
                    self.get_endeff_rot(), # end-effector rotation (quaternion)
                    self.get_endeff_vel(), # end-effector xyz velocity
                ])
            else:
                return np.concatenate([
                    self.get_endeff_pos(),
                    self.get_endeff_rot(),
                ])
        else:
            if self.keep_vel_in_obs:
                return np.concatenate([
                    self.get_endeff_pos(),
                    self.sim.data.qpos.flat,
                    self.sim.data.qvel.flat,
                ])
            else:
                return np.concatenate([
                    self.get_endeff_pos(),
                    self.sim.data.qpos.flat,
                ])

    def _get_obs_dict(self):
        ee_pos = self.get_endeff_pos()
        state_obs = self._get_env_obs()
        return dict(
            observation=state_obs,
            desired_goal=self._state_goal,
            achieved_goal=ee_pos,

            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=ee_pos,
        )

    def _convert_obs_dict_to_obs(self, obs_dict):
        return np.concatenate([obs_dict['observation'], obs_dict['desired_goal']])

    def _get_obs(self):
        return self._convert_obs_dict_to_obs(self._get_obs_dict())

    def _get_info(self):
        hand_distance = np.linalg.norm(self._state_goal - self.get_endeff_pos())
        return dict(
            hand_distance=hand_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
        )

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def get_endeff_rot(self):
        return self.data.body_xquat[self.endeff_id].copy()

    def get_endeff_vel(self):
        return self.data.body_xvelp[self.endeff_id].copy()

    def reset(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.init_angles
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        self.set_goal_xyz(self._state_goal)
        self.sim.forward()
        self.prev_qpos=self.data.qpos.copy()
        return self._get_obs()

    def set_goal_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[7:10] = pos.copy()
        qvel[7:10] = [0, 0, 0]
        self.set_state(qpos, qvel)

    @property
    def init_angles(self):
        return [
            1.02866769e+00, - 6.95207647e-01, 4.22932911e-01,
            1.76670458e+00, - 5.69637604e-01, 6.24117280e-01,
            3.53404635e+00,
            1.07586388e-02, 6.62018003e-01, 2.09936716e-02,
            1.00000000e+00, 3.76632959e-14, 1.36837913e-11, 1.56567415e-23
        ]

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    @property
    def goal_id(self):
        return self.model.body_names.index('goal')

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'hand_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    """
    Multitask functions
    """
    @property
    def goal_dim(self) -> int:
        return 3

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(batch_size, self.goal_space.low.size),
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        hand_pos = achieved_goals
        goals = desired_goals
        ctrl_cost = self.ctrl_cost_coef * np.sum(np.abs(actions), axis=1)

        distances = np.linalg.norm(hand_pos - goals, axis=1)
        if self.reward_type == 'hand_distance':
            r = -distances
        elif self.reward_type == 'hand_success':
            r = -(distances < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r - ctrl_cost

    def set_to_goal(self, goal):
        raise NotImplementedError()

    def get_env_state(self):
        joint_state = self.sim.get_state()
        goal = self._state_goal.copy()
        return joint_state, goal

    def set_env_state(self, state):
        state, goal = state
        self.sim.set_state(state)
        self.sim.forward()
        self._state_goal = goal

    def log_diagnostics(self, paths):
        diagnostics = self.get_diagnostics(paths)
        logger.record_tabular('HandDistanceMean', diagnostics['hand_distance Mean'])
        logger.record_tabular('FinalHandDistanceMean', diagnostics['Final hand_distance Mean'])
        logger.record_tabular('FinalHandSuccessMean', diagnostics['Final hand_success Mean'])

if __name__ == "__main__":
    H = 20000

    env = SawyerReachTorqueEnv(keep_vel_in_obs=False, use_safety_box=False)
    env.get_goal()
    # env = MultitaskToFlatEnv(env)
    lock_action = False
    while True:
        obs = env.reset()
        for i in range(H):
            #action = env.action_space.sample() / 100
            a = np.sin(i/2)
            action = np.asarray([-20, 1, 1, 1, 1, 1, -1])
            print(action)
            obs, reward, _, info = env.step(action)
            env.render()
        break
        print("new episode")