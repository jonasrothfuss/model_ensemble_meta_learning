from collections import OrderedDict
import numpy as np
from gym.spaces import Dict
from rllab.spaces import Box
from rllab.misc import logger

from sandbox.ours.envs.sawyer.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from sandbox.ours.envs.sawyer.core.multitask_env import MultitaskEnv
from sandbox.ours.envs.sawyer.base import SawyerXYZEnv
from rllab import config

GRIPPER_STATE_LIMIT = (0.0, 0.11) # distance between left and right gripper part

class SawyerPickAndPlaceEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            reward_type='hand_and_obj_distance',
            indicator_threshold=0.06,

            obj_init_pos=(0, 0.6, 0.02),

            fix_goal=True,
            fixed_goal=(0, 0.85, 0.02, 0, 0.85, 0.02), #3D placing goal, for hand and object
            height_target=0.1,
            goal_low=None,
            goal_high=None,

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
        if obj_low is None:
            obj_low = self.hand_low
        if obj_high is None:
            obj_high = self.hand_high

        if goal_low is None:
            goal_low = np.hstack((self.hand_low, obj_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, obj_high))

        self.max_path_length = 150

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.obj_init_pos = np.array(obj_init_pos)

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self.height_target = height_target

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

        self.observation_space = Box(
            np.hstack((self.hand_and_obj_space.low, np.array(GRIPPER_STATE_LIMIT[0]))),
            np.hstack((self.hand_and_obj_space.high, np.array(GRIPPER_STATE_LIMIT[1]))),
        )

        self.goal_space = Box(goal_low, goal_high)

        self._observation_space_dict = Dict([
            ('observation', self.observation_space),
            ('desired_goal', self.hand_and_obj_space),
            ('achieved_goal', self.hand_and_obj_space),
            ('state_observation', self.observation_space),
            ('state_desired_goal', self.hand_and_obj_space),
            ('state_achieved_goal', self.hand_and_obj_space),
        ])
        self.observation_space = Box(np.concatenate([self.observation_space.low, self.hand_and_obj_space.low]),
                                     np.concatenate([self.observation_space.high, self.hand_and_obj_space.high]))

        self.reset()

    @property
    def model_name(self):
        return 'sawyer_long_gripper/sawyer_pick_and_place.xml'

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0.5
        self.viewer.cam.lookat[2] = 0.1
        self.viewer.cam.distance = 1.4
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 310
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        obs_dict = self._get_obs_dict()
        obs = self._convert_obs_dict_to_obs(obs_dict)

        reward , pickRew, placeRew = self.compute_rewards(action, obs_dict) #TODO
        self.curr_path_length +=1

        info = self._get_info()
        info.update({'pickRew': pickRew, 'placeRew': placeRew})

        done = self.curr_path_length >= self.max_path_length

        return obs, reward, done, info

    def _get_obs_dict(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
        achieved_goal = np.concatenate((e, b))
        gripper_state = np.array([self.get_gripper_state()])
        observation = np.concatenate([achieved_goal, gripper_state])

        return dict(
            observation=observation,
            desired_goal=self._state_goal,
            achieved_goal=achieved_goal,
            state_observation=observation,
            state_desired_goal=self._state_goal,
            state_achieved_goal=achieved_goal,
        )

    def _convert_obs_dict_to_obs(self, obs_dict):
        return np.concatenate([obs_dict['observation'], obs_dict['desired_goal']])

    def _get_obs(self):
        return self._convert_obs_dict_to_obs(self._get_obs_dict())

    def _get_info(self):
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM = (rightFinger + leftFinger) / 2
        obj_pos = self.get_body_com("obj")

        obj_height = obj_pos[2]

        hand_goal = self._state_goal[:3]
        obj_goal = self._state_goal[3:]
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos())
        obj_distance = np.linalg.norm(obj_goal - obj_pos)
        touch_distance = np.linalg.norm(obj_pos - fingerCOM)
        return dict(
            obj_height=obj_height,
            hand_distance=hand_distance,
            obj_distance=obj_distance,
            touch_distance=touch_distance,
            obj_success=float(obj_distance < self.indicator_threshold),
            touch_success=float(touch_distance < self.indicator_threshold),
        )

    def get_obj_pos(self):
        return self.data.get_body_xpos('obj').copy()

    def get_gripper_state(self):
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        return np.linalg.norm(rightFinger - leftFinger)

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )
       
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('goal'), 2] = (
                -1000
            )

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        assert self._state_goal.shape == (6,)
        self._set_goal_marker(self._state_goal[-3:])

        self._set_obj_xyz(self.obj_init_pos)

        self.curr_path_length = 0
        self.pickCompleted = False

        init_obj = self.obj_init_pos

        heightTarget , placingGoal = self.height_target, self._state_goal[-3:]

        self.maxPlacingDist = np.linalg.norm([init_obj[0], init_obj[1], heightTarget] - placingGoal) + heightTarget

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def put_obj_in_hand(self):
        new_obj_pos = self.data.get_site_xpos('endeffector')
        new_obj_pos[1] -= 0.01
        self.do_simulation(-1)
        self.do_simulation(1)
        self._set_obj_xyz(new_obj_pos)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        hand_goal = state_goal[:3]
        for _ in range(30):
            self.data.set_mocap_pos('mocap', hand_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # keep gripper closed
            self.do_simulation(np.array([1]))
        self._set_obj_xyz(state_goal[3:])
        self.sim.forward()

    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(self.fixed_goal.copy()[None], batch_size, 0)
        else:
            goals = np.random.uniform(
                self.hand_and_obj_space.low,
                self.hand_and_obj_space.high,
                size=(batch_size, self.hand_and_obj_space.low.size),
            )
        return {
            'state_desired_goal': goals,
        }

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obs):
           
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
       
        heightTarget = self.height_target
        placingGoal = self._state_goal[-3:]

        objPos = self.get_body_com("obj")
        fingerCOM = (rightFinger + leftFinger)/2

        graspDist = np.linalg.norm(objPos - fingerCOM)
        graspRew = -graspDist

        grasp_attained = graspDist < 0.1

        tolerance = 0.01
        pick_completed = objPos[2] >= (heightTarget - tolerance)

        def pickReward():

            if pick_completed and grasp_attained:
                return 10*heightTarget

            elif (objPos[2]> 0.025) and grasp_attained:
                return 10*min(heightTarget, objPos[2])
         
            else:
                return 0

        def placeReward():
            placingDist = np.linalg.norm(objPos - placingGoal)

            if self.pickCompleted and grasp_attained:
                return max(100*(self.maxPlacingDist - placingDist),0)

            else:
                return 0 

        pickRew = pickReward()
        placeRew = placeReward()
        reward = graspRew + pickRew + placeRew

        #returned in a list because that's how compute_reward in multiTask.env expects it
        return [reward, pickRew, placeRew]

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'obj_height',
            'hand_distance',
            'obj_distance',
            'touch_distance',
            'obj_success',
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

    def log_diagnostics(self, paths):
        diagnostics = self.get_diagnostics(paths)

        logger.record_tabular('HandDistanceMean', diagnostics['hand_distance Mean'])
        logger.record_tabular('ObjectDistanceMean', diagnostics['obj_distance Mean'])
        logger.record_tabular('TouchDistanceMean', diagnostics['touch_distance Mean'])
        logger.record_tabular('ObjectHeightMax', diagnostics['obj_height Max'])

        logger.record_tabular('FinalHandDistanceMean', diagnostics['Final hand_distance Mean'])
        logger.record_tabular('FinalObjectDistanceMean', diagnostics['Final obj_distance Mean'])

        logger.record_tabular('FinalObjectSuccessMean', diagnostics['Final obj_success Mean'])
        logger.record_tabular('FinalHandAndObjSuccessMean', diagnostics['Final touch_success Mean'])

if __name__ == "__main__":
    env = SawyerPickAndPlaceEnv(fix_goal=False)
    import time
    env.reset()
    for _ in range(1000):
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())  # take a random action
        time.sleep(env.dt)