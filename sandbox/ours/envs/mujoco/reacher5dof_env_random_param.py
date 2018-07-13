from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from sandbox.ours.envs.mujoco.base_env_rand_param import BaseEnvRandParams
from sandbox.ours.envs.helpers import get_all_function_arguments
from rllab.misc import logger
import numpy as np
import warnings

class Reacher5DofEnvRandParams(BaseEnvRandParams, MujocoEnv, Serializable):
    FILE = 'reacher_5dof.xml'

    def __init__(self, *args, log_scale_limit=2.0, fix_params=False, fixed_goal=True, random_seed=None, **kwargs):
        self.sign = 1
        self.first = True
        self.penalty_ctrl = 0.01

        self.init_geom_size = None
        self.init_body_pos = None
        self.init_geom_pos = None


        args_all, kwargs_all = get_all_function_arguments(self.__init__, locals())
        BaseEnvRandParams.__init__(self, log_scale_limit=log_scale_limit, fix_params=fix_params,
                                   rand_params=['geom_size'], fixed_goal=fixed_goal, random_seed=random_seed)

        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.__init__(*args_all, **kwargs_all)

    def step(self, action):
        # distance between end of reacher and the goal
        vec = self.get_body_com("fingertip") - self.get_body_com("target")

        # calculate reward
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + self.penalty_ctrl * reward_ctrl

        # take a step
        self.forward_dynamics(action)
        obs = self.get_current_obs()
        done = False
        return Step(obs, float(reward), done, reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    @overrides
    def reset_mujoco(self, init_state=None, evaluating=None):

        if (init_state is None):

            # set random joint starting positions
            qpos = np.random.uniform(low=-1.5, high=1.5, size=self.init_qpos.shape) + self.init_qpos

            # set random goal position
            if (self.fixed_goal):
                self.goal = np.array([0.25, 0.15])
            else:
                while True:
                    potential_goal = np.random.uniform(low=-.15, high=.15, size=2)
                    if np.linalg.norm(potential_goal) > 0.03:
                        self.goal = potential_goal
                        break
            qpos[-2:, 0] = self.goal

            # set random starting joint velocities
            qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.init_qvel.shape)
            # set 0 vel for the goal
            qvel[-2:, 0] = 0

            # set vars
            self.model.data.qpos = qpos
            self.model.data.qvel = qvel
            self.model.data.qacc = self.init_qacc
            self.model.data.ctrl = self.init_ctrl

        else:
            start = 0
            for datum_name in ["qpos", "qvel", "qacc", "ctrl"]:
                datum = getattr(self.model.data, datum_name)
                datum_dim = datum.shape[0]
                datum = init_state[start: start + datum_dim]
                setattr(self.model.data, datum_name, datum)
                start += datum_dim

    def get_current_obs(self):
        # qpos 7 things (5pos, goalx, goaly)
        # qvel 7 things (5pos, goalx, goaly)
        # OBS: 5,5,2,5,2
        theta = self.model.data.qpos.flat[:-2]
        return np.concatenate([
            np.cos(theta), # joint angles
            np.sin(theta),
            self.model.data.qpos.flat[-2:], #end effector position
            self.model.data.qvel.flat[:-2], #velocities
            (self.get_body_com('fingertip') - self.get_body_com("target"))[:2] # distance from goal
        ])

    def get_reward(self, obs, next_obs, action):
        vec = next_obs[:, -2:]
        reward_dist = -np.linalg.norm(vec, axis=1)
        reward_ctrl = -np.sum(np.square(action), axis=1)
        reward = reward_dist + self.penalty_ctrl * reward_ctrl
        return reward

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        """
        resets the environment and returns the initial observation
        :param init_state: initial state -> joint angles
        :param reset_args: dicts containing a randomized parameter set for altering the mujoco model params
                         if None, a new set of model params is sampled
        :return: initial observation
        """

        if self.init_geom_size is None:
            self._save_initial_config()

        # The first time reset is called -> sample and fix the mujoco parameters
        if self.fix_params and not self.parameters_already_fixed:
            self.sample_and_fix_parameters()

        if self.fix_params and reset_args is not None:
            warnings.warn("Environment parameters are fixed - reset_ars does not have any effect", UserWarning)

        # set mujoco model parameters
        elif (not self.fix_params) and (reset_args is not None):
            self.reset_mujoco_parameters(reset_args)
        elif not self.fix_params:
            # sample parameter set
            reset_args = self.sample_env_params(1)[0]
            self.reset_mujoco_parameters(reset_args)

        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def sample_env_params(self, num_param_sets, log_scale_limit=None):
        """
              generates randomized parameter sets for the mujoco env
              :param num_param_sets: number of parameter sets to obtain
              :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
              :return: array of length num_param_sets with dicts containing a randomized parameter set
              """
        assert hasattr(self, 'random_state'), "random_state must be set in the constructor"

        if log_scale_limit is None:
            log_scale_limit = self.log_scale_limit

        param_sets=[]

        for _ in range(num_param_sets):

            new_params = {}

            geom_size = self.init_geom_size.copy()
            body_pos = self.init_body_pos.copy()
            geom_pos = self.init_geom_pos.copy()
            links = self.random_state.choice([0, 1, 2, 3, 4], size=2, replace=False)
            links.sort()
            for link in links:
                geom_idx = self.model.geom_names.index('link' + str(link))
                body_son_idx = self.model.body_names.index('body' + str(link)) + 1

                size_multiplier = np.array(2.0) ** self.random_state.uniform(-log_scale_limit, log_scale_limit)
                normal_size = self.init_geom_size[geom_idx][1]
                size = size_multiplier * normal_size
                geom_size[geom_idx][1] = size
                body_pos[body_son_idx][0] = 2 * size
                geom_pos[geom_idx][0] = size
            new_params['geom_size'] = geom_size
            new_params['body_pos'] = body_pos
            new_params['geom_pos'] = geom_pos

            param_sets.append(new_params)

        return param_sets

    def _save_initial_config(self):
        self.init_geom_size = self.model.geom_size.copy()
        self.init_body_pos = self.model.body_pos.copy()
        self.init_geom_pos = self.model.geom_pos.copy()

if __name__ == "__main__":

    env = Reacher5DofEnvRandParams(log_scale_limit=1.0, fix_params=False)
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())  # take a random action

    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())  # take a random action
