from rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.ours.envs.normalized_env import normalize
from sandbox.ours.envs.base import TfEnv
from rllab.misc.instrument import run_experiment_lite
from sandbox.ours.envs.sawyer import SawyerReachXYZEnv, SawyerPickAndPlaceEnv, SawyerReachTorqueEnv, SawyerPushAndReachXYZEnv
from rllab.misc.instrument import VariantGenerator
from rllab import config
from experiments.helpers.ec2_helpers import cheapest_subnets
from sandbox.ours.algos.MAML.maml_ppo import MAMLPPO
from sandbox.ours.policies.maml_gauss_mlp_policy import MAMLGaussianMLPPolicy

import tensorflow as tf
import sys
import argparse
import random
import numpy as np

EXP_PREFIX = 'sawyer-push-ppo-maml'

ec2_instance = 'c4.4xlarge'
subnets = cheapest_subnets(ec2_instance, num_subnets=3)

PUCK_GOAL_TARGET = np.array([-0.2, 0.65])
INIT_PUCK_TARGET = np.array([0.00, 0.60])

def run_train_task(vv):

    env = TfEnv(normalize(vv['env_class'](
        fix_goal=vv['fix_goal'],
        reward_type=vv['reward_type'],
        init_puck_low=INIT_PUCK_TARGET - vv['init_slack'],
        init_puck_high=INIT_PUCK_TARGET + vv['init_slack'],
        puck_goal_low= PUCK_GOAL_TARGET - vv['goal_slack'],
        puck_goal_high=PUCK_GOAL_TARGET + vv['goal_slack'],
    )))

    policy = MAMLGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        num_tasks=vv['meta_batch_size'],
        hidden_sizes=vv['hidden_sizes'],
        grad_step_size=vv['fast_lr'],
        hidden_nonlinearity=vv['hidden_nonlinearity'],
        trainable_step_size=vv['trainable_step_size'],
        bias_transform=vv['bias_transform']
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    optimizer_args = dict(
        max_epochs=vv['max_epochs'],
        batch_size=vv['num_batches'],
        tf_optimizer_args=dict(learning_rate=vv['outer_lr']),
    )

    algo = MAMLPPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=vv['fast_batch_size'],  # number of trajs for grad update
        max_path_length=vv['path_length'],
        meta_batch_size=vv['meta_batch_size'],
        num_grad_updates=vv['num_grad_updates'],
        n_itr=vv['n_itr'],
        discount=vv['discount'],
        entropy_bonus=vv['entropy_bonus'],
        clip_eps=vv['clip_eps'],
        clip_outer=vv['clip_outer'],
        target_outer_step=vv['target_outer_step'],
        target_inner_step=vv['target_inner_step'],
        init_outer_kl_penalty=vv['init_outer_kl_penalty'],
        init_inner_kl_penalty=vv['init_inner_kl_penalty'],
        adaptive_outer_kl_penalty=vv['adaptive_outer_kl_penalty'],
        adaptive_inner_kl_penalty=vv['adaptive_inner_kl_penalty'],
        parallel_sampler=vv['parallel_sampler'],
        optimizer_args=optimizer_args,
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
    vg.add('env', ['SawyerPushAndReachXYZEnv'])
    vg.add('fix_goal', [False])
    vg.add('goal_slack', [0.0, 0.05, 0.1])
    vg.add('init_slack', [0.0, 0.05])
    vg.add('reward_type', ['puck_distance_hand_distance_after_success'])

    vg.add('seed', [1, 10])
    vg.add('n_itr', [1001])
    vg.add('fast_lr', [0.1])
    vg.add('outer_lr', [1e-3])
    vg.add('meta_batch_size', [40])
    vg.add('num_grad_updates', [1])
    vg.add('fast_batch_size', [20])

    vg.add('discount', [0.99])
    vg.add('path_length', [200])
    vg.add('hidden_nonlinearity', ['tanh'])
    vg.add('hidden_sizes', [(64, 64)])
    vg.add('trainable_step_size', [False])
    vg.add('bias_transform', [False])
    vg.add('entropy_bonus', [0])

    # PPO-MAML params
    vg.add('clip_eps', [0.5])
    vg.add('clip_outer', [True])
    vg.add('target_outer_step', [0])
    vg.add('init_outer_kl_penalty', [0])
    vg.add('adaptive_outer_kl_penalty', [False])
    vg.add('target_inner_step', [1e-2])
    vg.add('init_inner_kl_penalty', [1e-3])
    vg.add('adaptive_inner_kl_penalty', [True])
    vg.add('max_epochs', [5])
    vg.add('num_batches', [1])
    vg.add('parallel_sampler', [True])

    variants = vg.variants()

    # ----------------------- AWS conficuration ---------------------------------
    if args.mode == 'ec2':
        info = config.INSTANCE_TYPE_INFO[ec2_instance]
        n_parallel = info['vCPU']
    else:
        n_parallel = 8

    if args.mode == 'ec2':


        config.AWS_INSTANCE_TYPE = ec2_instance
        config.AWS_SPOT_PRICE = str(info["price"])

        print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(variants)))
        print('Running on type {}, with price {}, on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                           config.AWS_SPOT_PRICE, ), str(subnets))

    # ----------------------- TRAINING ---------------------------------------
    exp_ids = random.sample(range(1, 1000), len(variants))
    for v, exp_id in zip(variants, exp_ids):
        exp_name = "%s_%s_%.1f_%.3f_%i_%i_id_%i" % (
            EXP_PREFIX, v['env'], v['clip_eps'], v['target_inner_step'], v['max_epochs'], v['seed'], exp_id)

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
            snapshot_mode="gap",
            snapshot_gap=200,
            periodic_sync=True,
            sync_s3_pkl=True,
            sync_s3_log=True,
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=v["seed"],
            #sync_all_data_node_to_s3=True,
            python_command="python3",
            pre_commands=["yes | pip install --upgrade pip",
                          "yes | pip install --upgrade cloudpickle"],
            mode=args.mode,
            use_cloudpickle=True,
            variant=v,
        )


def instantiate_class_stings(v):
    v['env_class'] = globals()[v['env']]
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