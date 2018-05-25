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
from sandbox.jonas.envs.mujoco import AntEnvRandParams, HalfCheetahEnvRandParams, HopperEnvRandParams

import tensorflow as tf
import sys
import argparse
import random

EXP_PREFIX = 'model-ensemble-trpo'

ec2_instance = 'c4.4xlarge'
subnets = cheapest_subnets(ec2_instance, num_subnets=3)


def run_train_task(vv):

    env = TfEnv(normalize(vv['env'](log_scale_limit=vv['log_scale_limit'])))

    dynamics_model = MLPDynamicsEnsemble(
        name="dyn_model",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes_model'],
        weight_normalization=vv['weight_normalization_model'],
        num_models=vv['num_models']
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
        dynamic_model_epochs=vv['dynamic_model_epochs'],
        num_gradient_steps_per_iter=vv['num_gradient_steps_per_iter'],
        max_path_length=vv['path_length'],
        n_itr=vv['n_itr'],
        discount=vv['discount'],
        step_size=vv["step_size"],
        reset_policy_std=vv['reset_policy_std'],
        reinit_model_cycle=vv['reinit_model_cycle']
    )
    algo.train()

def run_experiment(argv):

    # -------------------- Parse Arguments -----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')

    args = parser.parse_args(argv[1:])

    # -------------------- Define Variants -----------------------------------

    vg = VariantGenerator()
    vg.add('env', ['AntEnvRandParams']) # HalfCheetahEnvRandParams
    vg.add('n_itr', [100])
    vg.add('log_scale_limit', [0.0])
    vg.add('step_size', [0.01])
    vg.add('seed', [22, 33, 55]) #TODO set back to [1, 11, 21, 31, 41]
    vg.add('discount', [0.99])
    vg.add('path_length', [100])
    vg.add('batch_size_env_samples', [5000])
    vg.add('batch_size_dynamics_samples', [50000])
    vg.add('initial_random_samples', [5000])
    vg.add('dynamic_model_epochs', [(100, 50)])
    vg.add('num_gradient_steps_per_iter', [50]) #TODO change back to 50
    vg.add('hidden_nonlinearity_policy', ['tanh'])
    vg.add('hidden_nonlinearity_model', ['relu'])
    vg.add('hidden_sizes_policy', [(32, 32)])
    vg.add('hidden_sizes_model', [(1024, 1024)])
    vg.add('weight_normalization_model', [True])
    vg.add('retrain_model_when_reward_decreases', [True])
    vg.add('reset_policy_std', [True])
    vg.add('reinit_model_cycle', [0])
    vg.add('num_models', [5])
    variants = vg.variants()

    # ----------------------- AWS conficuration ---------------------------------
    if args.mode == 'ec2':
        info = config.INSTANCE_TYPE_INFO[ec2_instance]
        n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
    else:
        n_parallel = 6

    if args.mode == 'ec2':


        config.AWS_INSTANCE_TYPE = ec2_instance
        config.AWS_SPOT_PRICE = str(info["price"])

        print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(variants)))
        print('Running on type {}, with price {}, on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                           config.AWS_SPOT_PRICE, ), str(subnets))

    # ----------------------- TRAINING ---------------------------------------
    exp_ids = random.sample(range(1, 1000), len(variants))
    for v, exp_id in zip(variants, exp_ids):
        exp_name = "model_trpo_train_env_%s_%i_%i_%i_%i_id_%i" % (v['env'], v['path_length'], v['num_gradient_steps_per_iter'],
                                                               v['batch_size_env_samples'], v['seed'], exp_id)
        v = instantiate_class_stings(v)

        subnet = random.choice(subnets)
        config.AWS_REGION_NAME = subnet[:-1]
        config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
            config.AWS_REGION_NAME]
        config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
            config.AWS_REGION_NAME]
        config.AWS_SECURITY_GROUP_IDS = \
            config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
                config.AWS_REGION_NAME]


        run_experiment_lite(
            run_train_task,
            exp_prefix=EXP_PREFIX,
            exp_name=exp_name,
            # Number of parallel workers for sampling
            n_parallel=n_parallel,
            snapshot_mode="gap",
            snapshot_gap=5,
            periodic_sync=True,
            sync_s3_pkl=True,
            sync_s3_log=True,
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=v["seed"],
            python_command='python3',
            pre_commands=["yes | pip install tensorflow=='1.6.0'",
                          "pip list",
                          "yes | pip install --upgrade cloudpickle"],
            mode=args.mode,
            use_cloudpickle=True,
            variant=v,
        )


def instantiate_class_stings(v): #TODO change to
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