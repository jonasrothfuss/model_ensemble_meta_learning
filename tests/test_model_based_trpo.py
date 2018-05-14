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
from sandbox.jonas.algos.model_trpo import ModelTRPO

import tensorflow as tf


class TestMModelBasedTRPO(unittest.TestCase):

    def test_training(self):
        env = TfEnv(normalize(PointEnv()))

        tf.set_random_seed(22)
        np.random.seed(22)

        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(16, 16),
            hidden_nonlinearity=tf.nn.tanh
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        dynamics_model = MLPDynamicsModel("dyn_model", env, hidden_sizes=(16, 16))

        # fit dynamics model

        algo = ModelTRPO(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            baseline=baseline,
            batch_size_env_samples=5000,
            initial_random_samples=10000,
            batch_size_dynamics_samples=40000,
            max_path_length=100,
            dynamic_model_epochs=(30, 10),
            num_gradient_steps_per_iter=2,
            n_itr=20,
            discount=0.99,
            step_size=0.001,
        )
        algo.train()

if __name__ == '__main__':
    unittest.main()