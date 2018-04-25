from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import run_experiment_lite
from sandbox.jonas.envs.mujoco import HalfCheetahEnvRandParams, AntEnvRandParams, HopperEnvRandParams
import experiments.helpers.evaluation as eval
from rllab.misc.instrument import VariantGenerator, variant

import tensorflow as tf
import sys
import argparse
import random

EXP_PREFIX = 'trpo-and-param-env-baselines'

from pprint import pprint


def run_train_task(vv):
    env = TfEnv(normalize(vv['env'](log_scale_limit=vv["log_scale_limit"])))

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
        n_itr=vv['n_iter'],
        discount=vv['discount'],
        step_size=vv["step_size"],
    )
    algo.train()

def run_experiment(argv):

    # -------------------- Parse Arguments -----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')
    parser.add_argument('--n_parallel', type=int, default=1,
                        help='Number of parallel workers to perform rollouts. 0 => don\'t start any workers')

    args = parser.parse_args(argv[1:])


    # -------------------- Define Variants -----------------------------------

    vg = VariantGenerator()
    vg.add('env', ['HalfCheetahEnvRandParams'])
    vg.add('n_itr', [500])
    vg.add('log_scale_limit', [0.001, 0.01, 0.1, 1.0, 2.0])
    vg.add('step_size', [0.01, 0.05, 0.1]),
    vg.add('seed', [1, 11, 21, 31, 41])
    vg.add('discount', [0.99])
    vg.add('n_iter', [500])
    vg.add('path_length', [100])
    vg.add('batch_size', [20000])
    vg.add('hidden_nonlinearity', ['relu'])
    vg.add('hidden_sizes', [(64, 64)])

    variants = vg.variants()

    # ----------------------- TRAINING ---------------------------------------
    exp_ids = random.sample(range(1, 1000), len(variants))
    for v, exp_id in zip(variants, exp_ids):
        exp_name = "trpo_train_env_%s_%.3f_%.3f_%i_id_%i" % (v['env'], v['log_scale_limit'], v['step_size'], v['seed'], exp_id)
        v = instantiate_class_stings(v)

        run_experiment_lite(
            run_train_task,
            exp_prefix=EXP_PREFIX,
            exp_name=exp_name,
            # Number of parallel workers for sampling
            n_parallel=args.n_parallel,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=v["seed"],
            python_command=sys.executable,
            mode=args.mode,
            use_cloudpickle=True,
            variant=v,
            # plot=True,
            # terminate_machine=False,
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