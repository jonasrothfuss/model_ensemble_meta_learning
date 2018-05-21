from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy

from sandbox.rocky.tf.algos.trpo import TRPO
import tensorflow as tf
from rllab.envs.mujoco.half_cheetah_gym_env import HalfCheetahGymEnv
# from rllab.envs.mujoco.half_cheetah_env_rand import HalfCheetahEnvRand
# from rllab.envs.gym_env_initenv import GymEnvInit
# stub(globals())

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/rllab/issues/87#issuecomment-282519288
for task_num in range(1):
    env = TfEnv(HalfCheetahGymEnv())
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # grad_step_size=v['fast_lr'],
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(300, 300),
    )
    # load_discriminator = None #'/home/abhigupta/abhishek_sandbox/unsup_metalearning/sac/data/halfcheetah-longtrain/seed_20/itr_4920.pkl'

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        # load_policy=None,
        # discriminator=None,
        # load_discriminator=load_discriminator,
        # load_polvals=None,
        # task_num=task_num,
        baseline=baseline,
        batch_size=5000,
        max_path_length=100,
        n_itr=100,
        discount=0.99,
        step_size=0.01,
        # reset_arg=task_num,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    )
    # run_experiment_lite(
    #     algo.train,
    #     n_parallel=1,
    #     snapshot_mode="last",
    #     exp_name="TRPO_test_goalvel_cheetah_gym" + str(task_num) + 'seed' + str(20),
    #     exp_prefix="TRPO_test_goalvel_cheetah_gym",
    #     seed=20
    # )
    algo.train()