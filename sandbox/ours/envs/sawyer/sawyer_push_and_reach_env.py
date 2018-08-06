from collections import OrderedDict
import numpy as np
from gym.spaces import Dict
from rllab.spaces import Box
from rllab.misc import logger

from sandbox.ours.envs.sawyer.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from sandbox.ours.envs.sawyer.core.multitask_env import MultitaskEnv
from sandbox.ours.envs.sawyer.base import SawyerXYZEnv


class SawyerPushAndReachXYZEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            init_puck_low=[0.0, 0.6],
            init_puck_high=[0.0, 0.6],

            reward_type='hand_and_puck_distance',
            indicator_threshold=0.06,

            fix_goal=False,
            fixed_goal=(0.15, 0.6, 0.055, -0.15, 0.6),
            puck_goal_low=[-0.3, 0.55],
            puck_goal_high=[-0.1, 0.75],

            hide_goal_markers=False,

            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        if init_puck_low is None:
            init_puck_low = self.hand_low[:2]
        if init_puck_high is None:
            init_puck_high = self.hand_high[:2]
        init_puck_low = np.array(init_puck_low)
        init_puck_high = np.array(init_puck_high)

        self.puck_low = init_puck_low
        self.puck_high = init_puck_high

        """ Set goal sampling range """
        if puck_goal_low is None:
            goal_low = np.hstack((self.hand_low, self.hand_low[:2]))
        else:
            goal_low = np.hstack((self.hand_low, puck_goal_low))
        if puck_goal_high is None:
            goal_high = np.hstack((self.hand_high, self.hand_high[:2]))
        else:
            goal_high = np.hstack((self.hand_high, puck_goal_high))
        self.hand_and_puck_goal_space = Box(goal_low, goal_high)


        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        self.hand_and_puck_space = Box(
            np.hstack((self.hand_low, self.hand_low[0:2])),
            np.hstack((self.hand_high, self.hand_high[0:2])),
        )

        self.hand_space = Box(self.hand_low, self.hand_high)
        self._observation_space_dict = Dict([
            ('observation', self.hand_and_puck_space),
            ('desired_goal', self.hand_and_puck_space),
            ('achieved_goal', self.hand_and_puck_space),
            ('state_observation', self.hand_and_puck_space),
            ('state_desired_goal', self.hand_and_puck_space),
            ('state_achieved_goal', self.hand_and_puck_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', self.hand_space),
            ('proprio_achieved_goal', self.hand_space),
        ])
        self.init_puck_z = self.get_puck_pos()[2]
        self.observation_space = Box(np.concatenate([self.hand_and_puck_space.low, self.hand_and_puck_goal_space.low]),
                                     np.concatenate([self.hand_and_puck_space.high, self.hand_and_puck_goal_space.high]))


    @property
    def model_name(self):
        return 'sawyer_push_puck.xml'

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 1.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.3
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 270
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        self.set_xyz_action(action)
        # keep gripper closed
        self.do_simulation(np.array([1]))
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        obs_dict = self._get_obs_dict()
        obs = self._convert_obs_dict_to_obs(obs_dict)
        reward = self.compute_reward(action, obs_dict)
        info = self._get_info()
        done = False
        return obs, reward, done, info

    def _get_obs_dict(self):
        e = self.get_endeff_pos()
        b = self.get_puck_pos()[:2]
        flat_obs = np.concatenate((e, b))

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=flat_obs[:3],
            proprio_desired_goal=self._state_goal[:3],
            proprio_achieved_goal=flat_obs[:3],
        )

    def _convert_obs_dict_to_obs(self, obs_dict):
        return np.concatenate([obs_dict['observation'], obs_dict['desired_goal']])

    def _get_obs(self):
        return self._convert_obs_dict_to_obs(self._get_obs_dict())

    def _get_info(self):
        hand_goal = self._state_goal[:3]
        puck_goal = self._state_goal[3:]
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos())
        puck_distance = np.linalg.norm(puck_goal - self.get_puck_pos()[:2])
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_puck_pos()
        )
        return dict(
            hand_distance=hand_distance,
            puck_distance=puck_distance,
            hand_and_puck_distance=hand_distance+puck_distance,
            touch_distance=touch_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
            puck_success=float(puck_distance < self.indicator_threshold),
            hand_and_puck_success=float(
                hand_distance+puck_distance < self.indicator_threshold
            ),
            touch_success=float(touch_distance < self.indicator_threshold),
        )

    def get_puck_pos(self):
        return self.data.get_body_xpos('puck').copy()

    def sample_puck_xy(self):
        return np.random.uniform(low=self.puck_low, high=self.puck_high)

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal[:3]
        )
        self.data.site_xpos[self.model.site_name2id('puck-goal-site')][:2] = (
            goal[3:]
        )
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = (
                -1000
            )
            self.data.site_xpos[self.model.site_name2id('puck-goal-site'), 2] = (
                -1000
            )

    def _set_puck_xy(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[8:11] = np.hstack((pos.copy(), np.array([0.02])))
        qpos[11:15] = np.array([1, 0, 0, 0])
        qvel[8:15] = 0
        self.set_state(qpos, qvel)

    def reset(self, init_state=None, reset_args=None):
        return super().reset(init_state=init_state, reset_args=reset_args)

    def reset_model(self, init_state=None, reset_args=None):
        self._reset_hand()

        goal = self.sample_goal() if reset_args is None else reset_args
        self.reset_gaol(goal)

        init_state = self.sample_puck_xy() if init_state is None else init_state
        self._set_puck_xy(init_state)

        return self._get_obs()

    def reset_gaol(self, goal):
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)


    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.02]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_to_goal(self, goal):
        hand_goal = goal['state_desired_goal'][:3]
        puck_goal = goal['state_desired_goal'][3:]
        for _ in range(30):
            self.data.set_mocap_pos('mocap', hand_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # keep gripper closed
            self.do_simulation(np.array([1]))
        self._set_puck_xy(puck_goal)
        self.sim.forward()

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.hand_and_puck_goal_space.low,
                self.hand_and_puck_goal_space.high,
                size=(batch_size, self.hand_and_puck_space.low.size),
            )
        goal_array = [{'desired_goal': goal.flatten(), 'state_desired_goal': goal.flatten()}
                      for goal in np.vsplit(goals, batch_size)]
        return goal_array

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :3]
        puck_pos = achieved_goals[:, 3:]
        hand_goals = desired_goals[:, :3]
        puck_goals = desired_goals[:, 3:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, axis=1)
        puck_distances = np.linalg.norm(puck_goals - puck_pos, axis=1)
        hand_and_puck_distances = hand_distances + puck_distances
        puck_zs = self.init_puck_z * np.ones((desired_goals.shape[0], 1))
        touch_distances = np.linalg.norm(
            hand_pos - np.hstack((puck_pos, puck_zs)),
            axis=1,
        )
        puck_success = (puck_distances < self.indicator_threshold).astype(float)

        if self.reward_type == 'hand_distance':
            r = -hand_distances
        elif self.reward_type == 'hand_success':
            r = -(hand_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'puck_distance':
            r = -puck_distances
        elif self.reward_type == 'puck_success':
            r = -(puck_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'hand_and_puck_distance':
            r = -hand_and_puck_distances
        elif self.reward_type == 'hand_and_puck_distance_puck_success':
            r = -hand_and_puck_distances -(puck_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'hand_and_puck_success':
            r = -(hand_and_puck_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'touch_distance':
            r = -touch_distances
        elif self.reward_type == 'touch_success':
            r = -(touch_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'puck_distance_hand_distance_after_success':
            r = - puck_distances + puck_success * (4 - hand_distances) # give a sweet bonus if the puck is at it's goal
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'puck_distance',
            'hand_and_puck_distance',
            'touch_distance',
            'hand_success',
            'puck_success',
            'hand_and_puck_success',
            'touch_success',
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

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        self._set_goal_marker(goal)

    def log_diagnostics(self, paths, prefix=''):
        diagnostics = self.get_diagnostics(paths)

        logger.record_tabular(prefix+'HandDistanceMean', diagnostics['hand_distance Mean'])
        logger.record_tabular(prefix+'PuckDistanceMean', diagnostics['puck_distance Mean'])
        logger.record_tabular(prefix+'TouchDistanceMean', diagnostics['touch_distance Mean'])

        logger.record_tabular(prefix+'FinalHandDistanceMean', diagnostics['Final hand_distance Mean'])
        logger.record_tabular(prefix+'FinalPuckDistanceMean', diagnostics['Final puck_distance Mean'])

        logger.record_tabular(prefix+'FinalHandSuccessMean', diagnostics['Final hand_success Mean'])
        logger.record_tabular(prefix+'FinalPuckSuccessMean', diagnostics['Final puck_success Mean'])
        logger.record_tabular(prefix+'FinalHandAndPuckSuccessMean', diagnostics['Final hand_and_puck_success Mean'])


class SawyerPushAndReachXYEnv(SawyerPushAndReachXYZEnv):
    def __init__(self, *args, hand_z_position=0.055, **kwargs):
        self.quick_init(locals())
        SawyerPushAndReachXYZEnv.__init__(self, *args, **kwargs)
        self.hand_z_position = hand_z_position
        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.fixed_goal[2] = hand_z_position
        hand_and_puck_low = self.hand_and_puck_space.low.copy()
        hand_and_puck_low[2] = hand_z_position
        hand_and_puck_high = self.hand_and_puck_space.high.copy()
        hand_and_puck_high[2] = hand_z_position
        self.hand_and_puck_space = Box(hand_and_puck_low, hand_and_puck_high)
        self.observation_space = Dict([
            ('observation', self.hand_and_puck_space),
            ('desired_goal', self.hand_and_puck_space),
            ('achieved_goal', self.hand_and_puck_space),
            ('state_observation', self.hand_and_puck_space),
            ('state_desired_goal', self.hand_and_puck_space),
            ('state_achieved_goal', self.hand_and_puck_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', self.hand_space),
            ('proprio_achieved_goal', self.hand_space),
        ])

    def step(self, action):
        delta_z = self.hand_z_position - self.data.mocap_pos[0, 2]
        action = np.hstack((action, delta_z))
        return super().step(action)

if __name__ == "__main__":
    import time

    PUCK_GOAL_TARGET = np.array([-0.2, 0.65])
    INIT_PUCK_TARGET = np.array([0.00, 0.60])
    goal_slack = 0.001
    puck_slack = 0.001

    env = SawyerPushAndReachXYZEnv(
        fix_goal=False,
        init_puck_low=INIT_PUCK_TARGET - puck_slack,
        init_puck_high=INIT_PUCK_TARGET + puck_slack,
        puck_goal_low=PUCK_GOAL_TARGET - goal_slack,
        puck_goal_high=PUCK_GOAL_TARGET + goal_slack,
    )
    env.reset()
    for _ in range(1000):
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())  # take a random action
        time.sleep(env.dt)