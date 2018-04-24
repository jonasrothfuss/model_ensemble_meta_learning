from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import run_experiment_lite
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from experiments.envs.mujoco.half_cheetah_env_rand_param import HalfCheetahEnvRandParams
import sys

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):

    @variant
    def log_scale_limit(self):
        return [0.1, 1.0, 2.0]

    @variant
    def step_size(self):
        return [0.01, 0.05, 0.1]

    @variant
    def seed(self):
        return [1, 11, 21, 31, 41]


def run_task(vv):

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
        batch_size=4000,
        max_path_length=100,
        n_itr=500,
        discount=0.99,
        step_size=vv["step_size"],
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()


variants = VG().variants()

for v in variants:

    run_experiment_lite(
        run_task,
        exp_prefix="trpo_cheetah_rand_param_baselines",
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        mode="local",
        python_command=sys.executable,
        # mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
    sys.exit()
