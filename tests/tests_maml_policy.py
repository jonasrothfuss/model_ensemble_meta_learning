import unittest
from rllab.envs.own_envs.point_2d_env import PointEnv
from sandbox.jonas.controllers import RandomController
from sandbox.jonas.model_based_rl.helpers import sample, path_reward
from sandbox.jonas.dynamics import MLPDynamicsModel
import tensorflow as tf
import numpy as np
import pickle
import joblib
import os


from rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.jonas.policies.maml_improved_gauss_mlp_policy import MAMLImprovedGaussianMLPPolicy
from sandbox.jonas.envs.own_envs import PointEnvMAML
from sandbox_maml.rocky.tf.envs.base import TfEnv
from rllab_maml.envs.normalized_env import normalize
from sandbox.jonas.algos.MAML.maml_trpo import MAMLTRPO

class TestMAMLImprovedGaussPolicy(unittest.TestCase):

    def sample_random_trajectories_point_env(self, env, num_paths=100, horizon=100):
        env.reset()
        random_controller = RandomController(env)
        random_paths = sample(env, random_controller, num_paths=100, horizon=100)
        return random_paths

    def test_serialization(self):

        env = TfEnv(normalize(PointEnvMAML()))
        obs = env.reset()

        policy = MAMLImprovedGaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(16, 16),
            hidden_nonlinearity=tf.nn.tanh
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        import rllab.misc.logger as logger

        logger.set_snapshot_dir('/tmp/')
        logger.set_snapshot_mode('last')

        algo = MAMLTRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=2,
            max_path_length=10,
            meta_batch_size=4,
            num_grad_updates=1,
            n_itr=1,
            discount=0.99,
            step_size=0.01,
        )
        algo.train()

        tf.reset_default_graph()
        pkl_file = os.path.join('/tmp/','params.pkl')
        with tf.Session() as sess:
            data = joblib.load(pkl_file)
            policy = data['policy']
            action_before = policy.get_action(obs)[1]['mean']

            dump_string = pickle.dumps(policy)

        tf.reset_default_graph()
        with tf.Session() as sess:
            policy_loaded = pickle.loads(dump_string)
            action_after = policy_loaded.get_action(obs)[1]['mean']

        diff = np.sum(np.abs(action_before - action_after))
        self.assertAlmostEquals(diff, 0.0, places=3)


    def test_get_mean(self):

        env = TfEnv(normalize(PointEnvMAML()))
        obs = env.reset()

        policy = MAMLImprovedGaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(16, 16),
            hidden_nonlinearity=tf.nn.tanh,
            trainable_step_size=True,
            grad_step_size=0.7
        )


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            mean_stepsize_1 = policy.get_mean_step_size()

        self.assertAlmostEquals(mean_stepsize_1, 0.7, places=5)
