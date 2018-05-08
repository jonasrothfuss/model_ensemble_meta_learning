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

    def test_train_prediction(self):
        # just checks if training and prediction runs without errors and prediction returns correct shapes
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=10, horizon=10)
        dynamics_model = MLPDynamicsModel("dyn_model_2", env, hidden_sizes=(16, 16))

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=5)
            obs_pred = dynamics_model.predict(obs, act)
            self.assertEqual(obs_pred.shape, obs.shape)

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
        paths = sample_random_trajectories_point_env(env, num_paths=10, horizon=10)
        dynamics_model = MLPDynamicsEnsemble("dyn_ensemble_2", env, hidden_sizes=(16, 16), num_models=5)

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=5)
            obs_pred = dynamics_model.predict(obs, act, pred_type='mean')
            self.assertEqual(obs_pred.shape, obs.shape)

            obs_pred = dynamics_model.predict(obs, act, pred_type='rand')
            self.assertEqual(obs_pred.shape, obs.shape)

            obs_pred = dynamics_model.predict(obs, act, pred_type='all')
            self.assertEqual(obs_pred.shape, obs.shape + (5,))