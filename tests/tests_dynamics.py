import unittest
from rllab.envs.own_envs.point_2d_env import PointEnv
from sandbox.jonas.controllers import RandomController
from sandbox.jonas.model_based_rl.helpers import sample, path_reward
from sandbox.jonas.dynamics import MLPDynamicsModel, MLPDynamicsEnsemble
import tensorflow as tf
import numpy as np
import pickle



def sample_random_trajectories_point_env(env, num_paths=100, horizon=100):
    env.reset()
    random_controller = RandomController(env)
    random_paths = sample(env, random_controller, num_paths=100, horizon=100)
    return random_paths


class TestMLPDynamics(unittest.TestCase):

    def test_serialization(self):
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=10, horizon=10)
        dynamics_model = MLPDynamicsModel("dyn_model", env, hidden_sizes=(16,16))

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=5)
            obs_pred = dynamics_model.predict(obs, act)

            dump_string = pickle.dumps(dynamics_model)

        tf.reset_default_graph()
        with tf.Session() as sess:
            dynamics_model_loaded = pickle.loads(dump_string)
            #dynamics_model_loaded.fit(obs, act, obs_next, epochs=5)
            obs_pred_loaded = dynamics_model_loaded.predict(obs, act)

        diff = np.sum(np.abs(obs_pred_loaded - obs_pred))

        self.assertAlmostEquals(diff, 0, places=2)


    def test_train_prediction1(self):
        env = PointEnv()
        obs = np.random.uniform(-2, 2, size=(20000, 2))
        act = np.random.uniform(-0.1, 0.1, size=(20000, 2))
        next_obs = obs + act

        dynamics_model = MLPDynamicsModel("dyn_model_2a", env, hidden_sizes=(32, 32), normalize_input=False)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, next_obs, epochs=10, verbose=True)

            obs_test = np.random.uniform(-2, 2, size=(20000, 2))
            act_test = np.random.uniform(-0.1, 0.1, size=(20000, 2))
            obs_next_test = obs_test + act_test

            obs_next_pred = dynamics_model.predict(obs_test, act_test)
            mean_diff = np.mean(np.abs(obs_next_test - obs_next_pred))
            print("Mean Diff:", mean_diff)

            self.assertEqual(obs_next_pred.shape, obs_test.shape)
            self.assertLessEqual(mean_diff, 0.01)

    def test_train_prediction2(self):
        # just checks if training and prediction runs without errors and prediction returns correct shapes

        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=500, horizon=100)
        dynamics_model = MLPDynamicsModel("dyn_model_2b", env, hidden_sizes=(32, 32), normalize_input=True)

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        obs_test = np.random.uniform(-2, 2, size=(20000, 2))
        act_test = np.random.uniform(-0.1, 0.1, size=(20000, 2))
        obs_next_test = obs_test + act_test

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=20, verbose=True)
            obs_next_pred = dynamics_model.predict(obs_test, act_test)

            mean_diff = np.mean(np.abs(obs_next_test - obs_next_pred))
            print("Mean Diff:", mean_diff)

            self.assertEqual(obs_next_pred.shape, obs_test.shape)
            self.assertLessEqual(mean_diff, 0.01)

    def test_train_prediction_performance(self):
        # just checks if training and prediction runs without errors and prediction returns correct shapes
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=500, horizon=500)
        dynamics_model = MLPDynamicsModel("dyn_model_3", env, hidden_sizes=(16, 16))

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        paths_test = sample_random_trajectories_point_env(env, num_paths=10, horizon=100)
        obs_test = np.concatenate([path['observations'] for path in paths_test], axis=0)
        obs_next_test = np.concatenate([path['next_observations'] for path in paths_test], axis=0)
        act_test = np.concatenate([path['actions'] for path in paths_test], axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=20)

            next_obs_pred = dynamics_model.predict(obs_test, act_test)
            diff = np.mean(np.abs(next_obs_pred-obs_next_test)**2)
            print("DIFF:", diff)
            self.assertLess(diff, 0.05)


class TestMLPDynamicsEnsemble(unittest.TestCase):

    def test_serialization(self):
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=10, horizon=10)
        dynamics_model = MLPDynamicsEnsemble("dyn_ensemble_1", env, hidden_sizes=(16, 16))

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=5)
            obs_pred = dynamics_model.predict(obs, act, pred_type='mean')

            dump_string = pickle.dumps(dynamics_model)

        tf.reset_default_graph()
        with tf.Session() as sess:
            dynamics_model_loaded = pickle.loads(dump_string)
            # dynamics_model_loaded.fit(obs, act, obs_next, epochs=5)
            obs_pred_loaded = dynamics_model_loaded.predict(obs, act, pred_type='mean')

        diff = np.sum(np.abs(obs_pred_loaded - obs_pred))

        self.assertAlmostEquals(diff, 0, places=2)

    def test_train_prediction(self):
        # just checks if training and prediction runs without errors and prediction returns correct shapes
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=200, horizon=100)
        dynamics_model = MLPDynamicsEnsemble("dyn_ensemble_2", env, hidden_sizes=(16, 16), num_models=5)

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        paths_test = sample_random_trajectories_point_env(env, num_paths=10, horizon=100)
        obs_test = np.concatenate([path['observations'] for path in paths_test], axis=0)
        obs_next_test = np.concatenate([path['next_observations'] for path in paths_test], axis=0)
        act_test = np.concatenate([path['actions'] for path in paths_test], axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=10)

            obs_pred1 = dynamics_model.predict(obs_test, act_test, pred_type='mean')
            diff1 = np.mean(np.abs(obs_pred1 - obs_next_test) ** 2)
            self.assertEqual(obs_pred1.shape, obs.shape)
            self.assertLess(diff1, 0.01)

            obs_pred2 = dynamics_model.predict(obs_test, act_test, pred_type='rand')
            diff2 = np.mean(np.abs(obs_pred2 - obs_next_test) ** 2)
            self.assertEqual(obs_pred2.shape, obs.shape)
            self.assertLess(diff2, 0.01)

            obs_pred3 = dynamics_model.predict(obs_test, act_test, pred_type='all')
            self.assertEqual(obs_pred3.shape, obs.shape + (5,))

    def test_train_prediction_std(self):
        # just checks if std prediction returns correct shapes
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=10, horizon=10)
        dynamics_model = MLPDynamicsEnsemble("dyn_ensemble_3", env, hidden_sizes=(16, 16), num_models=5)

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=5)
            std = dynamics_model.predict_std(obs, act)
            self.assertEqual(std.shape, obs.shape)