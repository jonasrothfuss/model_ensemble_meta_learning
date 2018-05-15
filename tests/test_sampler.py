import unittest
from rllab.envs.own_envs.point_2d_env import PointEnv
from sandbox.jonas.controllers import RandomController
from sandbox.jonas.model_based_rl.helpers import sample
from sandbox.jonas.dynamics import MLPDynamicsModel, MLPDynamicsEnsemble
import numpy as np

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.jonas.policies.maml_improved_gauss_mlp_policy import MAMLImprovedGaussianMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO

from sandbox.jonas.sampler.MAML_model_sampler.maml_model_vectorized_sampler import MAMLModelVectorizedSampler
from sandbox.jonas.sampler.model_sampler.model_vectorized_sampler import ModelVectorizedSampler
from sandbox.jonas.sampler.random_vectorized_sampler import RandomVectorizedSampler

import tensorflow as tf


from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

def sample_random_trajectories_point_env(env, num_paths=100, horizon=100):
    env.reset()
    random_controller = RandomController(env)
    random_paths = sample(env, random_controller, num_paths=num_paths, horizon=horizon)
    return random_paths

class TestModelSampler(unittest.TestCase):

    def test_policy_sampling(self):
        # get from data
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=100, horizon=100)
        dynamics_model = MLPDynamicsModel("dyn_model", env, hidden_sizes=(16,16))

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        env = TfEnv(normalize(PointEnv()))

        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(16, 16),
            hidden_nonlinearity=tf.nn.tanh
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        # fit dynamics model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=5)

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=20000,
                max_path_length=100,
                n_itr=10,
                discount=0.99,
                step_size=0.01,
            )

            algo.dynamics_model = dynamics_model

            itr = 1

            model_sampler = ModelVectorizedSampler(algo)
            model_sampler.start_worker()
            paths = model_sampler.obtain_samples(itr)
            samples_data = model_sampler.process_samples(itr, paths)

            print(samples_data.keys())

    def test_random_sampling(self):
        # get from data
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=100, horizon=100)
        dynamics_model = MLPDynamicsModel("dyn_model", env, hidden_sizes=(16,16))

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        env = TfEnv(normalize(PointEnv()))

        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(16, 16),
            hidden_nonlinearity=tf.nn.tanh
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        # fit dynamics model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=5)

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=20000,
                max_path_length=100,
                n_itr=10,
                discount=0.99,
                step_size=0.01,
            )

            algo.dynamics_model = dynamics_model

            itr = 1

            random_sampler = RandomVectorizedSampler(algo)
            random_sampler.start_worker()
            paths = random_sampler.obtain_samples(itr)
            samples_data = random_sampler.process_samples(itr, paths)

            self.assertTrue(set(samples_data.keys()) ==
                            set(['actions_dynamics', 'next_observations_dynamics', 'observations_dynamics']))

    def test_maml_sampling(self):
        # get from data
        # get from data
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=100, horizon=100)
        dynamics_model = MLPDynamicsEnsemble("dyn_model", env, hidden_sizes=(16,16), num_models=4)

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        env = TfEnv(normalize(PointEnv()))

        policy = MAMLImprovedGaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(100, 100),
            grad_step_size=0.1,
            hidden_nonlinearity=tf.nn.tanh,
            trainable_step_size=False,
            bias_transform=False
        )

        from rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        # fit dynamics model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            dynamics_model.fit(obs, act, obs_next, epochs=1)

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=20000,
                max_path_length=100,
                n_itr=10,
                discount=0.99,
                step_size=0.01,
            )

            algo.batch_size_dynamics_samples = algo.batch_size

            algo.dynamics_model = dynamics_model

            itr = 1

            model_sampler = MAMLModelVectorizedSampler(algo)
            model_sampler.start_worker()
            paths = model_sampler.obtain_samples(itr, return_dict=True)
            samples_data = model_sampler.process_samples(itr, paths[0])

            print(samples_data.keys())


if __name__ == '__main__':
    unittest.main()