from rllab.misc.instrument import VariantGenerator
from rllab import config
from rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab_maml.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.jonas.envs.normalized_env import normalize
from sandbox.jonas.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.jonas.policies.maml_improved_gauss_mlp_policy import MAMLImprovedGaussianMLPPolicy
from sandbox.jonas.dynamics.dynamics_ensemble import MLPDynamicsEnsemble
from sandbox.ignasi.dynamics.probabilistic_dynamics_ensemble import MLPProbabilisticDynamicsEnsemble
from sandbox.jonas.algos.ModelMAML.model_maml_trpo import ModelMAMLTRPO
from experiments.helpers.ec2_helpers import cheapest_subnets
from rllab import config

from sandbox.jonas.envs.own_envs import PointEnvMAML
from sandbox.jonas.envs.mujoco import AntEnvRandParams, HalfCheetahEnvRandParams, HopperEnvRandParams
from sandbox.jonas.envs.mujoco import Reacher5DofEnvRandParams
from sandbox.jonas.envs.mujoco.cheetah_env import HalfCheetahEnv


import tensorflow as tf
import sys
import argparse
import random
import json
import os


def run_train_task(vv):
    import sys
    print(vv['exp_prefix'])
    sysout_log_path = os.path.join(config.LOG_DIR, 'local', vv['exp_prefix'], vv['exp_name'], 'stdout.log')
    sysout_log_file = open(sysout_log_path, 'w')
    sys.stdout = sysout_log_file

    env = TfEnv(normalize(vv['env'](log_scale_limit=vv['log_scale_limit'])))

    dynamics_model = vv['dynamics_model'](
        name="dyn_model",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes_model'],
        weight_normalization=vv['weight_normalization_model'],
        num_models=vv['num_models'],
        optimizer=vv['optimizer_model']
    )

    policy = MAMLImprovedGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes_policy'],
        hidden_nonlinearity=vv['hidden_nonlinearity_policy'],
        grad_step_size=vv['fast_lr'],
        trainable_step_size=vv['trainable_step_size'],
        bias_transform=vv['bias_transform']
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = ModelMAMLTRPO(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        baseline=baseline,
        n_itr=vv['n_itr'],
        n_iter=vv['n_itr'],
        batch_size_env_samples=vv['batch_size_env_samples'],
        batch_size_dynamics_samples=vv['batch_size_dynamics_samples'],
        initial_random_samples=vv['initial_random_samples'],
        dynamic_model_epochs=vv['dynamic_model_epochs'],
        num_maml_steps_per_iter=vv['num_maml_steps_per_iter'],
        max_path_length_env=vv['path_length_env'],
        max_path_length_dyn=vv.get('path_length_dyn', None),
        reset_from_env_traj=vv.get('reset_from_env_traj', False),
        discount=vv['discount'],
        step_size=vv["meta_step_size"],
        num_grad_updates=1,
        retrain_model_when_reward_decreases=vv['retrain_model_when_reward_decreases'],
        reset_policy_std=vv['reset_policy_std'],
        reinit_model_cycle=vv['reinit_model_cycle'],
        frac_gpu=vv.get('frac_gpu', 1),
    )
    algo.train()

    sysout_log_file.close()


def run_experiment(vargs):

    # ----------------------- TRAINING ---------------------------------------
    kwargs = json.load(open(vargs[1], 'r'))
    exp_id = random.sample(range(1, 1000), 1)[0]
    v = kwargs['variant']
    exp_name = "model_ensemble_maml_train_env_%s_%i_%i_%i_%i_id_%i" % (
                v['env'], v['path_length_env'], v['num_maml_steps_per_iter'],
                v['batch_size_env_samples'], v['seed'], exp_id)
    v = instantiate_class_stings(v)
    kwargs['variant'] = v

    run_experiment_lite(
        run_train_task,
        exp_name=exp_name,
        **kwargs
    )


def instantiate_class_stings(v):
    v['env'] = globals()[v['env']]
    v['dynamics_model'] = globals()[v['dynamics_model']]

    if 'nm_mbs_envs' in v.keys():
        v['num_models'] = v['nm_mbs_envs'][0]
        v['meta_batch_size'] = v['nm_mbs_envs'][1]
        v['batch_size_env_samples'] = v['nm_mbs_envs'][0]

    # optimizer
    if v['optimizer_model'] == 'sgd':
        v['optimizer_model'] = tf.train.GradientDescentOptimizer
    elif v['optimizer_model'] == 'adam':
        v['optimizer_model'] = tf.train.AdamOptimizer
    elif v['optimizer_model'] == 'momentum':
        v['optimizer_model'] = tf.train.MomentumOptimizer

    # nonlinearlity
    for nonlinearity_key in ['hidden_nonlinearity_policy', 'hidden_nonlinearity_model']:
        if v[nonlinearity_key] == 'relu':
            v[nonlinearity_key] = tf.nn.relu
        elif v[nonlinearity_key] == 'tanh':
            v[nonlinearity_key] = tf.tanh
        elif v[nonlinearity_key] == 'elu':
            v[nonlinearity_key] = tf.nn.elu
        else:
            raise NotImplementedError('Not able to recognize spicified hidden_nonlinearity: %s' % v['hidden_nonlinearity'])
    return v


if __name__ == "__main__":
    run_experiment(sys.argv)