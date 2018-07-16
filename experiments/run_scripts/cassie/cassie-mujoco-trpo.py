from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import run_experiment_lite
from sandbox.ours.envs.mujoco import CassieEnv
from rllab.misc.instrument import VariantGenerator
from rllab import config
from experiments.helpers.ec2_helpers import cheapest_subnets

import tensorflow as tf
import sys
import argparse
import random

EXP_PREFIX = 'cassie-trpo'

ec2_instance = 'c4.2xlarge'
subnets = cheapest_subnets(ec2_instance, num_subnets=3)


def run_train_task(vv):

    env = TfEnv(normalize(CassieEnv()))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes'],
        hidden_nonlinearity=vv['hidden_nonlinearity']
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=vv['batch_size'],
        max_path_length=vv['path_length'],
        n_itr=vv['n_itr'],
        discount=vv['discount'],
        step_size=vv["step_size"],
        force_batch_sampler=True
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
    vg.add('n_itr', [2000])
    vg.add('step_size', [0.01, 0.02])
    vg.add('seed', [1, 11, 21])
    vg.add('discount', [0.99])
    vg.add('path_length', [500])
    vg.add('batch_size', [50000])
    vg.add('hidden_nonlinearity', ['tanh'])
    vg.add('hidden_sizes', [(32, 32), (64, 64)])

    variants = vg.variants()

    # ----------------------- AWS conficuration ---------------------------------
    if args.mode == 'ec2':
        info = config.INSTANCE_TYPE_INFO[ec2_instance]
        n_parallel = info['vCPU']
    else:
        n_parallel = 12

    if args.mode == 'ec2':


        config.AWS_INSTANCE_TYPE = ec2_instance
        config.AWS_SPOT_PRICE = str(info["price"])

        print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(variants)))
        print('Running on type {}, with price {}, on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                           config.AWS_SPOT_PRICE, ), str(subnets))

    # ----------------------- TRAINING ---------------------------------------
    exp_ids = random.sample(range(1, 1000), len(variants))
    for v, exp_id in zip(variants, exp_ids):
        exp_name = "trpo_train_cassie_mujoco_%.3f_%i_%i_id_%i" % (v['step_size'], v['batch_size'], v['seed'], exp_id)
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
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            periodic_sync=True,
            sync_s3_pkl=True,
            sync_s3_log=True,
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=v["seed"],
            #sync_all_data_node_to_s3=True,
            python_command="python3",
            pre_commands=["yes | pip install --upgrade pip",
                          "yes | pip install tensorflow=='1.6.0'",
                          "yes | pip install --upgrade cloudpickle"],
            mode=args.mode,
            use_cloudpickle=True,
            variant=v,
        )


def instantiate_class_stings(v):
    if v['hidden_nonlinearity'] == 'relu':
        v['hidden_nonlinearity'] = tf.nn.relu
    elif v['hidden_nonlinearity'] == 'tanh':
        v['hidden_nonlinearity'] = tf.tanh
    elif v['hidden_nonlinearity'] == 'elu':
        v['hidden_nonlinearity'] = tf.nn.elu
    else:
        raise NotImplementedError('Not able to recognize spicified hidden_nonlinearity: %s' % v['hidden_nonlinearity'])
    return v


if __name__ == "__main__":
    run_experiment(sys.argv)