from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.jonas.envs.normalized_env import normalize
from sandbox.jonas.envs.base import TfEnv
from sandbox.jonas.policies.improved_gauss_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator
from rllab import config
from experiments.helpers.ec2_helpers import cheapest_subnets
from sandbox.jonas.dynamics import MLPDynamicsEnsemble
from sandbox.jonas.algos.ModelTRPO.model_trpo import ModelTRPO
from sandbox.jonas.envs.mujoco import AntEnvRandParams, HalfCheetahEnvRandParams, HopperEnvRandParams, SwimmerEnvRandParams, WalkerEnvRandomParams



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

    dynamics_model = MLPDynamicsEnsemble(
        name="dyn_model",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes_model'],
        weight_normalization=vv['weight_normalization_model'],
        num_models=vv['num_models'],
        valid_split_ratio=vv['valid_split_ratio'],
        rolling_average_persitency=vv['rolling_average_persitency']
    )

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes_policy'],
        hidden_nonlinearity=vv['hidden_nonlinearity_policy'],
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = ModelTRPO(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        baseline=baseline,
        batch_size_env_samples=vv['batch_size_env_samples'],
        batch_size_dynamics_samples=vv['batch_size_dynamics_samples'],
        initial_random_samples=vv['initial_random_samples'],
        num_gradient_steps_per_iter=vv['num_gradient_steps_per_iter'],
        max_path_length=vv['path_length'],
        n_itr=vv['n_itr'],
        retrain_model_when_reward_decreases=vv['retrain_model_when_reward_decreases'],
        discount=vv['discount'],
        step_size=vv["step_size"],
        reset_policy_std=vv['reset_policy_std'],
        reinit_model_cycle=vv['reinit_model_cycle']
    )
    algo.train()

    sysout_log_file.close()


def run_experiment(vargs):

    # ----------------------- TRAINING ---------------------------------------
    kwargs = json.load(open(vargs[1], 'r'))
    exp_id = random.sample(range(1, 1000), 1)[0]
    v = kwargs['variant']
    exp_name = "model_trpo_train_env_%s_%i_%i_%i_%i_id_%i" % (
                v['env'], v['path_length'], v['num_gradient_steps_per_iter'],
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
