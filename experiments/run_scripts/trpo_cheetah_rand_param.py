from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import run_experiment_lite
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from experiments.envs.mujoco.half_cheetah_env_rand_param import HalfCheetahEnvRandParams
import experiments.helpers.evaluation as eval
from rllab.misc.instrument import VariantGenerator, variant
import rllab.config as config

import sys

EXP_PREFIX = 'trpo-cheetah-rand-aram-baselines'

class VG(VariantGenerator):

    @variant
    def log_scale_limit(self):
        return [0.01, 0.1, 1.0, 2.0]

    @variant
    def step_size(self):
        return [0.01, 0.05, 0.1]

    @variant
    def seed(self):
        return [1, 11, 21, 31, 41]


def run_train_task(vv):

    env = TfEnv(normalize(HalfCheetahEnvRandParams(log_scale_limit=vv["log_scale_limit"])))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100, 100),
        name="policy"
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1000,
        max_path_length=100,
        n_itr=500,
        discount=0.99,
        step_size=vv["step_size"],
    )
    algo.train()


def run_eval_task(vv):
    # create environment and fix the mujoco parameters
    # TODO: adjust log_scale limit of environment
    env = TfEnv(normalize(HalfCheetahEnvRandParams(log_scale_limit=vv["log_scale_limit"],
                                                   random_seed=vv['env_param_seed']).sample_and_fix_parameters()))

    # load policy and baseline- Warning: resets the tf graph
    # also returns the tensorflow session which must be used in the further code
    policy, baseline, sess = eval.load_policy_and_baseline(vv)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1000,
        max_path_length=100,
        n_itr=20,
        discount=0.99,
        step_size=vv["step_size"],
    )
    algo.train(sess=sess)


variants = VG().variants()

# ----------------------- TRAINING ---------------------------------------

for v in variants:

    exp_name = EXP_PREFIX + "_%.3f_%.3f_%i"%(v['log_scale_limit'], v['step_size'], v['seed'])

    run_experiment_lite(
        run_train_task,
        exp_prefix=EXP_PREFIX,
        exp_name=exp_name,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        python_command=sys.executable,
        mode="local",
        use_cloudpickle=True,
        # mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )

# ----------------------- EVALUATION ---------------------------------------

for v in variants:
    exp_name = EXP_PREFIX + "_%.3f_%.3f_%i" % (v['log_scale_limit'], v['step_size'], v['seed'])
    exp_log_dir = eval.get_local_exp_log_dir(EXP_PREFIX, exp_name)
    print(exp_log_dir)
    eval.evaluate_policy_transfer(run_eval_task, exp_log_dir, num_samped_envs=5)