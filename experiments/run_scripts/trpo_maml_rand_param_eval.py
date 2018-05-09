import experiments.helpers.evaluation as eval

from rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab_maml.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab_maml.envs.normalized_env import normalize
from rllab_maml.misc.instrument import stub, run_experiment_lite
from sandbox_maml.rocky.tf.algos.vpg import VPG
from sandbox_maml.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox_maml.rocky.tf.envs.base import TfEnv
from experiments.helpers.ec2_helpers import cheapest_subnets

from rllab import config

import tensorflow as tf
import sys
import argparse
import os
import random
import pickle

EXP_PREFIX = 'trpo-maml-rand-param-env-eval'

ec2_instance = 'm4.large'


def run_eval_task(vv):

    # load policy and baseline- Warning: resets the tf graph
    # also returns the tensorflow session which must be used in the further code
    baseline, env, params_pickle_file = eval.load_baseline_and_env(vv)

    tf.reset_default_graph()

    # fix the mujoco parameters
    env_class = eval.get_env_class(env)
    env = TfEnv(normalize(env_class(log_scale_limit=vv["log_scale_limit"], fix_params=True,
                                             random_seed=vv['env_param_seed'])))


    step_size = vv['fast_lr']

    policy = None

    algo = VPG(
            env=env,
            policy=policy,
            load_policy=params_pickle_file,
            baseline=baseline,
            batch_size=20000,
            max_path_length=100,
            n_itr=5,
            optimizer_args={'init_learning_rate': step_size, 'tf_optimizer_args': {'learning_rate': step_size}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
        )
    algo.train()

def run_evaluation(argv):

    # -------------------- Parse Arguments -----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_prefix_dir', type=str, help='path to dump dir which contains folders with '
                                                         'the train results i.e. params.pkl and variant.json file')
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')
    parser.add_argument('--n_parallel', type=int, default=1,
                        help='Number of parallel workers to perform rollouts. 0 => don\'t start any workers')
    parser.add_argument('--num_sampled_envs', type=int, default=5,
                        help='number or environments with samples parameters')

    args = parser.parse_args(argv[1:])

    # ----------------------- EVALUATION ---------------------------------------

    exp_prefix = os.path.basename(args.exp_prefix_dir)
    eval_exp_prefix = exp_prefix + '-eval'
    evaluation_runs = eval.prepare_evaluation_runs(args.exp_prefix_dir, EXP_PREFIX, num_sampled_envs=args.num_sampled_envs)

    # ----------------------- AWS conficuration ---------------------------------
    if args.mode == 'ec2':
        subnets = cheapest_subnets(ec2_instance, num_subnets=3)
        info = config.INSTANCE_TYPE_INFO[ec2_instance]
        config.AWS_INSTANCE_TYPE = ec2_instance
        config.AWS_SPOT_PRICE = str(info["price"])

        print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(evaluation_runs)))
        print('Running on type {}, with price {}, on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                           config.AWS_SPOT_PRICE, ), str(subnets))

    for eval_exp_name, v in evaluation_runs:

        if args.mode == 'ec2':
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
            run_eval_task,
            exp_prefix=eval_exp_prefix,
            exp_name=eval_exp_name,
            # Number of parallel workers for sampling
            n_parallel=args.n_parallel,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=v["seed"],
            python_command="python3", #TODO
            mode=args.mode,
            use_cloudpickle=True,
            periodic_sync=True,
            variant=v,
            # plot=True,
            # terminate_machine=False,
        )



if __name__ == "__main__":
    run_evaluation(sys.argv)