from sandbox.jonas.envs.mujoco import AntEnvMAMLRandParams, HalfCheetahMAMLEnvRandParams, HopperEnvMAMLRandParams
from rllab.misc.instrument import VariantGenerator
from rllab import config

from sandbox_maml.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab_maml.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab_maml.envs.normalized_env import normalize
from rllab_maml.misc.instrument import stub, run_experiment_lite
from sandbox_maml.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox_maml.rocky.tf.envs.base import TfEnv

import tensorflow as tf
import sys
import argparse
import random

EXP_PREFIX = 'trpo-and-param-env-baselines'

ec2_instance = 'm4.4xlarge' #TODO: maybe choose a more performant instance

# configure instan

subnets = [
    'us-east-2a',
    'us-east-2b',
    'us-east-2c',
]

info = config.INSTANCE_TYPE_INFO[ec2_instance]
config.AWS_INSTANCE_TYPE = ec2_instance
config.AWS_SPOT_PRICE = str(info["price"])


def run_train_task(vv):
    env = TfEnv(normalize(vv['env'](log_scale_limit=vv["log_scale_limit"])))

    policy = MAMLGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes'],
        grad_step_size=vv['fast_lr'],
        hidden_nonlinearity=vv['hidden_nonlinearity']
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = MAMLTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=vv['fast_batch_size'], # number of trajs for grad update
        max_path_length=vv['path_length'],
        meta_batch_size=vv['meta_batch_size'],
        num_grad_updates=vv['num_grad_updates'],
        n_itr=vv['n_iter'],
        discount=vv['discount'],
        step_size=vv["meta_step_size"],
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
    vg.add('env', ['AntEnvMAMLRandParams'])
    vg.add('n_itr', [500])
    vg.add('log_scale_limit', [0.001, 0.01, 0.1, 1.0])

    vg.add('fast_lr', [0.1])
    vg.add('meta_batch_size', [50])
    vg.add('num_grad_updates', [1])
    vg.add('meta_step_size', [0.01, 0.02])
    vg.add('fast_batch_size', [20])
    vg.add('seed', [1, 11, 21, 31, 41])
    vg.add('discount', [0.99])
    vg.add('n_iter', [500])
    vg.add('path_length', [100])
    vg.add('batch_size', [50000])
    vg.add('hidden_nonlinearity', ['relu'])
    vg.add('hidden_sizes', [(32, 32)])

    variants = vg.variants()
    from pprint import pprint
    pprint(variants)

    # ----------------------- AWS conficuration ---------------------------------
    if args.mode == 'ec2':
        n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
    else:
        n_parallel = 6

    if args.mode == 'ecs':
        print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(variants)))
        print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                       config.AWS_SPOT_PRICE,
                                                                                       n_parallel), *subnets)

    # ----------------------- TRAINING ---------------------------------------
    exp_ids = random.sample(range(1, 1000), len(variants))
    for v, exp_id in zip(variants, exp_ids):
        exp_name = "trpo_maml_train_env_%s_%.3f_%.3f_%i_id_%i" % (v['env'], v['log_scale_limit'], v['meta_step_size'], v['seed'], exp_id)
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
            sync_s3_pkl=True,
            periodic_sync=True,
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=v["seed"],
            #sync_all_data_node_to_s3=True,
            python_command="python3",
            mode=args.mode,
            use_cloudpickle=True,
            variant=v,
        )


def instantiate_class_stings(v):
    v['env'] = globals()[v['env']]
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