import unittest
from rllab.envs.own_envs.point_2d_env import PointEnv
from sandbox.jonas.controllers import RandomController
from sandbox.jonas.model_based_rl.helpers import sample, path_reward
from sandbox.jonas.dynamics import MLPDynamicsModel, MLPDynamicsEnsemble
import tensorflow as tf
import numpy as np
import pickle

from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO

from sandbox.jonas.sampler.model_vec_env_executor import ModelVecEnvExecutor
from sandbox.jonas.sampler.model_vectorized_sampler import ModelVectorizedSampler

import tensorflow as tf


from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

def sample_random_trajectories_point_env(env, num_paths=100, horizon=100):
    env.reset()
    random_controller = RandomController(env)
    random_paths = sample(env, random_controller, num_paths=num_paths, horizon=horizon)
    return random_paths

class TestModelSampler(unittest.TestCase):

    def test_sampling(self):
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

if __name__ == '__main__':
    unittest.main()