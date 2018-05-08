import unittest
from rllab.envs.own_envs.point_2d_env import PointEnv
from sandbox.jonas.controllers import RandomController
from sandbox.jonas.model_based_rl.helpers import sample, path_reward
from sandbox.jonas.dynamics import MLPDynamicsModel
import tensorflow as tf
import numpy as np
import pickle
import joblib

from sandbox.jonas.envs.mujoco import HalfCheetahMAMLEnvRandParams
from sandbox_maml.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab_maml.envs.normalized_env import normalize
from sandbox.jonas.policies.maml_improved_gauss_mlp_policy import MAMLImprovedGaussianMLPPolicy
from sandbox_maml.rocky.tf.envs.base import TfEnv
from experiments.helpers.ec2_helpers import cheapest_subnets


class TestMAMLImprovedGaussPolicy(unittest.TestCase):

    def sample_random_trajectories_point_env(self, env, num_paths=100, horizon=100):
        env.reset()
        random_controller = RandomController(env)
        random_paths = sample(env, random_controller, num_paths=100, horizon=100)
        return random_paths

    def test_serialization(self):
        env = HalfCheetahMAMLEnvRandParams(log_scale_limit=0.1)
        obs = env.reset()
        pkl_file = '/home/jonasrothfuss/Dropbox/Eigene_Dateien/UC_Berkley/2_Code/model_ensemble_meta_learning/data/s3/trpo-maml-rand-param-env/trpo_maml_train_env_HalfCheetahMAMLEnvRandParams_0.050_0.010_1_id_234/params.pkl'
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
        print(diff)

