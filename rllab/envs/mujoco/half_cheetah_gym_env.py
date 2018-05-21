import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
from gym import spaces
from gym import utils
import gym.envs.mujoco 
import pickle
import gym
import gym.wrappers
import gym.envs
import gym.spaces
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product

num_tasks = 1000
# goal_vels = np.random.uniform(-2.0, 2.0, (num_tasks, ))
# import pickle
# pickle.dump(goal_vels, open('all_goal_vels.pkl','wb'))
# import IPython
# IPython.embed()

def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    else:
        raise NotImplementedError

class HalfCheetahGymEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, choice=0):
        self.choice = choice
        self.goal_vels = pickle.load(open('all_goal_vels.pkl','rb'))
        mujoco_env.MujocoEnv.__init__(self, '/home/ignasi/GitRepos/rllab-private/vendor/mujoco_models/half_cheetah.xml', 1)
        utils.EzPickle.__init__(self)
        self.observation_space = convert_gym_space(self.observation_space)
        self.action_space = convert_gym_space(self.action_space)

    def sample_goals(self, num_goals):
        return np.array(range(num_goals))

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs()
        if self.action_space is not None:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        run_cost = 1.*np.abs(self.get_body_comvel("torso")[0] - self.goal_vels[0])
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return next_obs, reward, done, dict()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.01, high=.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self, reset_args=None):
        choice = reset_args
        if choice is not None:
            self.choice = choice
        elif self.choice is None:
            self.choice = np.random.randint(num_tasks)
        observation = self.reset_model()
        return observation

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    @property
    def horizon(self):
        return 100

    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        return Step(next_obs, reward, done, **info)

    def log_diagnostics(self, paths, prefix=''):
        pass