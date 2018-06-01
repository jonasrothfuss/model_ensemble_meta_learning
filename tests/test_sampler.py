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
from sandbox.jonas.algos.ModelMAML.model_maml_trpo import ModelMAMLTRPO
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

class DummyDynamicsEnsemble():

    def __init__(self,
                 name,
                 env_spec,
                 num_models=5,
                 ):
        self.name = name
        self.env_spec = env_spec
        self.num_models = num_models

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env_spec.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env_spec.action_space.shape[0]

    def fit(self, obs, act, obs_next, epochs=50, compute_normalization=True, verbose=False):
        pass

    def predict_model_batches(self, obs_batches, act_batches):
        """
            Predict the batch of next observations for each model given the batch of current observations and actions for each model
            :param obs_batches: observation batches for each model concatenated along axis 0 - numpy array of shape (batch_size_per_model * num_models, ndim_obs)
            :param act_batches: action batches for each model concatenated along axis 0 - numpy array of shape (batch_size_per_model * num_models, ndim_act)
            :return: pred_obs_next_batch: predicted batch of next observations -
                                    shape:  (batch_size_per_model * num_models, ndim_obs)
        """
        assert obs_batches.shape[0] == act_batches.shape[0] and obs_batches.shape[0] % self.num_models == 0
        assert obs_batches.ndim == 2 and obs_batches.shape[1] == self.obs_space_dims
        assert act_batches.ndim == 2 and act_batches.shape[1] == self.action_space_dims

        obs_batches_original = obs_batches

        batch_size_per_model = obs_batches.shape[0] // self.num_models

        delta_batches = np.concatenate([np.ones((batch_size_per_model, self.obs_space_dims)) * 0.01 * i for i in range(self.num_models)], axis=0)

        pred_obs_batches = delta_batches
        assert pred_obs_batches.shape == obs_batches.shape
        return pred_obs_batches

class DummyEnv(PointEnv):

    def reset(self):
        return np.array([1.0, 1.0])

    def done(self, obs):
        if obs.ndim == 1:
            return np.random.uniform() < 0.2
        elif obs.ndim == 2:
            return np.random.uniform(size=obs.shape[0]) < 0.2



class TestModelSampler(unittest.TestCase):

    def test_policy_sampling(self):
        # get from data
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=100, horizon=100)
        dynamics_model = MLPDynamicsEnsemble("dyn_model1", env, hidden_sizes=(16,16))

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

            algo = ModelMAMLTRPO(
                env=env,
                dynamics_model=dynamics_model,
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
        dynamics_model = MLPDynamicsModel("dyn_model2", env, hidden_sizes=(16,16))

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        env = TfEnv(normalize(PointEnv()))

        policy = GaussianMLPPolicy(
            name="policy2",
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

            self.assertTrue(set(samples_data.keys()) >= set(['actions_dynamics', 'next_observations_dynamics', 'observations_dynamics']))

    def test_maml_sampling(self):
        # get from data
        # get from data
        env = PointEnv()
        paths = sample_random_trajectories_point_env(env, num_paths=100, horizon=100)
        dynamics_model = MLPDynamicsEnsemble("dyn_model3", env, hidden_sizes=(16,16), num_models=4)

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in paths], axis=0)
        act = np.concatenate([path['actions'] for path in paths], axis=0)

        env = TfEnv(normalize(PointEnv()))

        policy = MAMLImprovedGaussianMLPPolicy(
            name="policy3",
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
            algo.meta_batch_size = dynamics_model.num_models

            algo.batch_size_dynamics_samples = algo.batch_size

            algo.dynamics_model = dynamics_model

            itr = 1

            model_sampler = MAMLModelVectorizedSampler(algo)
            model_sampler.start_worker()
            paths = model_sampler.obtain_samples(itr, return_dict=True)
            samples_data = model_sampler.process_samples(itr, paths[0])

            print(samples_data.keys())

    def test_model_sampling_with_dummy(self):
        env = DummyEnv()
        dynamics_dummy = DummyDynamicsEnsemble("dyn_model4", env, num_models=4)
        env = TfEnv(normalize(DummyEnv()))

        policy = MAMLImprovedGaussianMLPPolicy(
            name="policy4",
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
            algo.meta_batch_size = dynamics_dummy.num_models

            algo.batch_size_dynamics_samples = algo.batch_size
            algo.dynamics_model = dynamics_dummy

            itr = 1

            model_sampler = MAMLModelVectorizedSampler(algo)
            model_sampler.start_worker()
            paths = model_sampler.obtain_samples(itr, return_dict=True)

            n_steps_per_model = np.array([np.sum([path['observations'].shape[0] for path in model_paths]) for model_paths in paths.values()])

            self.assertTrue(all(np.abs(n_steps_per_model - algo.batch_size//dynamics_dummy.num_models) <= algo.max_path_length))

            for i in range(dynamics_dummy.num_models):
                for path in paths[i]:
                    self.assertTrue((np.logical_or(path['observations'] == 1.0, path['observations'] == i*0.01)).all())

    def test_model_sampling_with_dummy_different_meta_batch_size(self):
        env = DummyEnv()
        dynamics_dummy = DummyDynamicsEnsemble("dyn_model4", env, num_models=4)
        env = TfEnv(normalize(DummyEnv()))

        policy = MAMLImprovedGaussianMLPPolicy(
            name="policy4",
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
            algo.meta_batch_size = dynamics_dummy.num_models * 2

            algo.batch_size_dynamics_samples = algo.batch_size
            algo.dynamics_model = dynamics_dummy

            itr = 1

            model_sampler = MAMLModelVectorizedSampler(algo)
            model_sampler.start_worker()
            paths = model_sampler.obtain_samples(itr, return_dict=True)

            n_steps_per_model = np.array(
                [np.sum([path['observations'].shape[0] for path in model_paths]) for model_paths in paths.values()])

            self.assertTrue(
                all(np.abs(n_steps_per_model - algo.batch_size // algo.meta_batch_size) <= algo.max_path_length))

            for i in range(dynamics_dummy.num_models):
                for path in paths[i]:
                    self.assertTrue(
                        (np.logical_or(path['observations'] == 1.0, path['observations'] == i//2 * 0.01)).all())

    def test_model_sampling_with_given_traj_starting_obs(self):
        env = DummyEnv()
        dynamics_dummy = DummyDynamicsEnsemble("dyn_model4", env, num_models=4)
        env = TfEnv(normalize(DummyEnv()))

        policy = MAMLImprovedGaussianMLPPolicy(
            name="policy4",
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
            algo.meta_batch_size = dynamics_dummy.num_models * 2

            algo.batch_size_dynamics_samples = algo.batch_size
            algo.dynamics_model = dynamics_dummy

            itr = 1

            model_sampler = MAMLModelVectorizedSampler(algo)
            model_sampler.start_worker()

            traj_starting_obs = np.array([[-1, -1],[-0.5, -0.5]])
            paths = model_sampler.obtain_samples(itr, return_dict=True, traj_starting_obs=traj_starting_obs)

            for i in range(dynamics_dummy.num_models):
                for path in paths[i]:
                    print(path['observations'][0])
                    print(np.abs(np.mean(path['observations'][0]) + 1.0) < 0.001)
                    self.assertTrue(
                        np.abs(np.mean(path['observations'][0]) + 1.0) < 0.001 or np.abs(np.mean(path['observations'][0])+0.5) < 0.001)




if __name__ == '__main__':
    unittest.main()