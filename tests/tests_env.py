import unittest
from sandbox.jonas.envs.mujoco.reacher_env_random_param import Reacher5DofEnvRandParams
from sandbox.jonas.envs.mujoco import AntEnvRandParams, HalfCheetahEnvRandParams, HopperEnvRandParams, SwimmerEnvRandParams, SnakeEnvRandParams, WalkerEnvRandomParams

import numpy as np
import pickle


class TestRandParamEnv(unittest.TestCase):

    def test_serialization(self):
        envs = [
            HalfCheetahEnvRandParams(log_scale_limit=0.2, fix_params=True, random_seed=22),
            HopperEnvRandParams(log_scale_limit=0.2, fix_params=True, random_seed=24),
            AntEnvRandParams(log_scale_limit=0.2, fix_params=True, random_seed=24)
        ]

        for env in envs:
            env.reset()
            geom_size_before = env.model.geom_size
            for _ in range(10):
                env.step(env.action_space.sample())  # take a random action

            # pickle and unpickle
            env = pickle.loads(pickle.dumps(env))

            env.reset()
            geom_size_after = env.model.geom_size

            diff = np.sum(np.abs(geom_size_before - geom_size_after))

            self.assertAlmostEquals(diff, 0, places=3)
            self.assertAlmostEquals(env.log_scale_limit, 0.2)

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

class TestHopperEnv(unittest.TestCase):

    def test_reward_fn(self):
        env = HopperEnvRandParams()
        obs = env.reset()
        rewards = []
        reward_ests = []
        actions = []
        obses = []
        for i in range(2):
            for _ in range(1000):
                #env.render()
                action = env.action_space.sample()
                actions.append(action)
                obs_new, reward, done, _ = env.step(action)
                reward_est = env.reward(obs, action, obs_new)
                rewards.append(reward)
                reward_ests.append(reward_est)
                obses.append(obs)
                obs = obs_new

        self.assertLessEqual(np.abs(np.sum(rewards) - np.sum(reward_ests)), 2.0)

        actions = np.stack(actions, axis=0)
        obses = np.stack(obses, axis=0)

        reward_est2 = env.reward(obses[:-1], actions[:-1], obses[1:])
        diff = np.sum(np.abs(reward_est2 - reward_ests[:-1]))
        self.assertAlmostEquals(diff, 0.0)

    def test_done_fn(self):
        env = HopperEnvRandParams()
        obs = env.reset()
        obses = []
        dones = []
        for i in range(2):
            for _ in range(1000):
                action = env.action_space.sample()
                obs_new, reward, done, _ = env.step(action)
                self.assertEquals(done, env.done(obs_new))
                dones.append(done)
                obses.append(obs)
                obs = obs_new

        obses = np.stack(obses, axis=0)

        dones2 = env.done(obses[1:])
        self.assertFalse(np.logical_xor(dones[:-1], dones2).any())

class TestCheetahEnv(unittest.TestCase):

    def test_reward_fn(self):
        env = HalfCheetahEnvRandParams()
        obs = env.reset()
        rewards = []
        reward_ests = []
        actions = []
        obses = []
        for i in range(2):
            for _ in range(1000):
                action = env.action_space.sample()
                actions.append(action)
                obs_new, reward, done, _ = env.step(action)
                reward_est = env.reward(obs, action, obs_new)
                rewards.append(reward)
                reward_ests.append(reward_est)
                obses.append(obs)
                obs = obs_new

        self.assertLessEqual(np.abs(np.sum(rewards) - np.sum(reward_ests)), 2.0)

        actions = np.stack(actions, axis=0)
        obses = np.stack(obses, axis=0)

        reward_est2 = env.reward(obses[:-1], actions[:-1], obses[1:])
        diff = np.sum(np.abs(reward_est2 - reward_ests[:-1]))
        self.assertAlmostEquals(diff, 0.0)

class TestAntEnv(unittest.TestCase):

    def test_reward_fn(self):
        env = AntEnvRandParams()
        obs = env.reset()
        rewards = []
        reward_ests = []
        actions = []
        obses = []
        for i in range(2):
            for _ in range(1000):
                action = env.action_space.sample()
                actions.append(action)
                obs_new, reward, done, _ = env.step(action)
                reward_est = env.reward(obs, action, obs_new)
                rewards.append(reward)
                reward_ests.append(reward_est)
                obses.append(obs)
                obs = obs_new

        print(np.corrcoef(rewards, reward_ests))
        self.assertLessEqual(np.abs(np.sum(rewards) - np.sum(reward_ests)), 10.0)

        actions = np.stack(actions, axis=0)
        obses = np.stack(obses, axis=0)

        # reward_est2 = env.reward(obses[:-1], actions[:-1], obses[1:])
        # diff = np.mean(np.abs(reward_est2 - reward_ests[:-1]))
        # self.assertAlmostEquals(diff, 0.0)

    def test_done_fn(self):
        env = AntEnvRandParams()
        obs = env.reset()
        obses = []
        dones = []
        for i in range(2):
            for _ in range(1000):
                action = env.action_space.sample()
                obs_new, reward, done, _ = env.step(action)
                self.assertEquals(done, env.done(obs_new))
                dones.append(done)
                obses.append(obs)
                obs = obs_new

        obses = np.stack(obses, axis=0)

        dones2 = env.done(obses[1:])
        self.assertFalse(np.logical_xor(dones[:-1], dones2).any())

class TestSwimmerEnv(unittest.TestCase):

    def test_reward_fn(self):
        env = SwimmerEnvRandParams()
        obs = env.reset()
        rewards = []
        reward_ests = []
        actions = []
        obses = []
        for i in range(2):
            for _ in range(1000):
                action = env.action_space.sample()
                actions.append(action)
                obs_new, reward, done, _ = env.step(action)
                reward_est = env.reward(obs, action, obs_new)
                #print("True reward", reward)
                #print("Estimated reward", reward_est)
                rewards.append(reward)
                reward_ests.append(reward_est)
                obses.append(obs)
                obs = obs_new

        print(np.corrcoef(rewards, reward_ests))
        print(np.mean(rewards) / np.mean(reward_ests))
        self.assertLessEqual(np.abs(np.sum(rewards) - np.sum(reward_ests)), 2.0)

        actions = np.stack(actions, axis=0)
        obses = np.stack(obses, axis=0)

        reward_est2 = env.reward(obses[:-1], actions[:-1], obses[1:])
        diff = np.sum(np.abs(reward_est2 - reward_ests[:-1]))
        self.assertAlmostEquals(diff, 0.0)

class TestSnakeEnv(unittest.TestCase):

    def test_reward_fn(self):
        env = SnakeEnvRandParams()
        obs = env.reset()
        rewards = []
        reward_ests = []
        actions = []
        obses = []
        for i in range(2):
            for _ in range(1000):
                action = env.action_space.sample()
                actions.append(action)
                obs_new, reward, done, _ = env.step(action)
                reward_est = env.reward(obs, action, obs_new)
                print("True reward", reward)
                print("Estimated reward", reward_est)
                rewards.append(reward)
                reward_ests.append(reward_est)
                obses.append(obs)
                obs = obs_new

        print(np.corrcoef(rewards, reward_ests))
        print(np.mean(rewards) / np.mean(reward_ests))
        self.assertLessEqual(np.abs(np.sum(rewards) - np.sum(reward_ests)), 2.0)

        actions = np.stack(actions, axis=0)
        obses = np.stack(obses, axis=0)

        reward_est2 = env.reward(obses[:-1], actions[:-1], obses[1:])
        diff = np.sum(np.abs(reward_est2 - reward_ests[:-1]))
        self.assertAlmostEqual(diff, 0.0)

class TestWalkerEnv(unittest.TestCase):

    def test_reward_fn(self):
        env = WalkerEnvRandomParams()
        obs = env.reset()
        rewards = []
        reward_ests = []
        actions = []
        obses = []
        for i in range(2):
            for _ in range(1000):
                action = env.action_space.sample()
                actions.append(action)
                obs_new, reward, done, _ = env.step(action)
                reward_est = env.reward(obs, action, obs_new)
                print("True reward", reward)
                print("Estimated reward", reward_est)
                rewards.append(reward)
                reward_ests.append(reward_est)
                obses.append(obs)
                obs = obs_new

        print(np.corrcoef(rewards, reward_ests))
        print(np.mean(rewards) / np.mean(reward_ests))
        self.assertLessEqual(np.abs(np.sum(rewards) - np.sum(reward_ests)), 2.0)

        actions = np.stack(actions, axis=0)
        obses = np.stack(obses, axis=0)

        reward_est2 = env.reward(obses[:-1], actions[:-1], obses[1:])
        diff = np.sum(np.abs(reward_est2 - reward_ests[:-1]))
        self.assertAlmostEqual(diff, 0.0)

    def test_done_fn(self):
        env = WalkerEnvRandomParams()
        obs = env.reset()
        obses = []
        dones = []
        for i in range(2):
            for _ in range(1000):
                action = env.action_space.sample()
                obs_new, reward, done, _ = env.step(action)
                self.assertEquals(done, env.done(obs_new))
                dones.append(done)
                obses.append(obs)
                obs = obs_new

        obses = np.stack(obses, axis=0)

        dones2 = env.done(obses[1:])
        self.assertFalse(np.logical_xor(dones[:-1], dones2).any())


if __name__ == '__main__':
    unittest.main()