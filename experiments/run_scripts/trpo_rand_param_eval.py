from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import run_experiment_lite
from sandbox.jonas.envs.mujoco import HalfCheetahEnvRandParams, AntEnvRandParams, HopperEnvRandParams
import experiments.helpers.evaluation as eval
from rllab.misc.instrument import VariantGenerator, variant

import sys
import argparse
import os

EXP_PREFIX = 'trpo-and-param-env-baselines'


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

def run_eval_task(vv):

    # load policy and baseline- Warning: resets the tf graph
    # also returns the tensorflow session which must be used in the further code
    policy, baseline, env, sess = eval.load_saved_objects(vv)

    # fix the mujoco parameters
    env_class = eval.get_env_class(env)
    env = TfEnv(normalize(env_class(log_scale_limit=vv["log_scale_limit"],
                                             random_seed=vv['env_param_seed']).sample_and_fix_parameters()))
    # TODO: maybe adjust log_scale limit of environment

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=vv['batch_size'],
        max_path_length=vv['path_length'],
        n_itr=10,
        discount=vv['discount'],
        step_size=vv["step_size"],
    )
    algo.train(sess=sess)


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

    eval_exp_prefix = os.path.basename(args.exp_prefix_dir) + '-eval'
    evaluation_runs = eval.prepare_evaluation_runs(args.exp_prefix_dir)

    for eval_exp_name, v in evaluation_runs:

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
            python_command=sys.executable,
            mode=args.mode,
            use_cloudpickle=True,
            variant=v,
            # plot=True,
            # terminate_machine=False,
        )



if __name__ == "__main__":
    run_evaluation(sys.argv)