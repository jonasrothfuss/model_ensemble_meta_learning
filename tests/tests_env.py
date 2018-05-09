import unittest
from sandbox.jonas.envs.mujoco.reacher_env_random_param import Reacher5DofEnvRandParams
import numpy as np
import pickle

class TestReacherEnvRandomParam(unittest.TestCase):

    def test_param_fixing_1(self):

        env = Reacher5DofEnvRandParams(log_scale_limit=1.0, fix_params=True)
        env.reset()
        geom_size_before = env.model.geom_size
        for _ in range(10):
            env.step(env.action_space.sample())  # take a random action

        env.reset()
        geom_size_after = env.model.geom_size

        diff = np.sum(np.abs(geom_size_before-geom_size_after))

        self.assertAlmostEquals(diff, 0, places=3)

    def test_param_fixing_2(self):

        env = Reacher5DofEnvRandParams(log_scale_limit=1.0, fix_params=False)
        env.reset()
        geom_size_before = env.model.geom_size
        for _ in range(10):
            env.step(env.action_space.sample())  # take a random action

        env.reset()
        geom_size_after = env.model.geom_size

        diff = np.sum(np.abs(geom_size_before-geom_size_after))

        self.assertGreater(diff, 0.01)

    def test_seed(self):
        env = Reacher5DofEnvRandParams(log_scale_limit=1.0, fix_params=False, random_seed=22)
        env.reset()
        geom_size_1= env.model.geom_size
        for _ in range(10):
            env.step(env.action_space.sample())  # take a random action

        # same seed
        env = Reacher5DofEnvRandParams(log_scale_limit=1.0, fix_params=False, random_seed=22)
        env.reset()
        geom_size_2 = env.model.geom_size
        for _ in range(10):
            env.step(env.action_space.sample())  # take a random action

        diff = np.sum(np.abs(geom_size_1 - geom_size_2))

        self.assertAlmostEquals(diff, 0, places=3)

        # different seed
        env = Reacher5DofEnvRandParams(log_scale_limit=1.0, fix_params=False, random_seed=2)
        env.reset()
        geom_size_2 = env.model.geom_size
        for _ in range(10):
            env.step(env.action_space.sample())  # take a random action

        diff = np.sum(np.abs(geom_size_1 - geom_size_2))

        self.assertGreater(diff, 0.01)

    def test_serialization(self):
        env = Reacher5DofEnvRandParams(log_scale_limit=0.2, fix_params=True, random_seed=22, fixed_goal=False)

        env.reset()
        geom_size_before = env.model.geom_size
        for _ in range(10):
            env.step(env.action_space.sample())  # take a random action

        #pickle and unpickle
        env = pickle.loads(pickle.dumps(env))


        env.reset()
        geom_size_after = env.model.geom_size

        diff = np.sum(np.abs(geom_size_before - geom_size_after))

        self.assertAlmostEquals(diff, 0, places=3)
        self.assertAlmostEquals(env.log_scale_limit, 0.2)
        self.assertFalse(env.fixed_goal)

    def test_fix_goal_1(self):
        env = Reacher5DofEnvRandParams(log_scale_limit=1.0, fix_params=True, random_seed=22, fixed_goal=True)
        env.reset()
        goal_1 = env.goal

        env.reset()
        goal_2 = env.goal

        diff = np.sum(np.abs(goal_1 - goal_2))
        self.assertAlmostEquals(diff, 0, places=3)

    def test_fix_goal_2(self):
        env = Reacher5DofEnvRandParams(log_scale_limit=1.0, fix_params=True, random_seed=22, fixed_goal=False)
        env.reset()
        goal_1 = env.goal

        env.reset()
        goal_2 = env.goal

        diff = np.sum(np.abs(goal_1 - goal_2))
        self.assertGreater(diff, 0.01)