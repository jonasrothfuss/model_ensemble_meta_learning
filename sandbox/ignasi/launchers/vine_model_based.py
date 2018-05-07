from sandbox.ignasi.algos.trpo import VINETRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.ignasi.envs.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.ignasi.envs.model_env import ModelEnv
from sandbox.ignasi.dynamics.mlp_dynamics import MLPDynamicsModel
from rllab.misc.instrument import stub, run_experiment_lite

# TODO: Watchout the actions in the real environment are normalized


real_env = TfEnv(normalize(HalfCheetahEnv()))

dynamics_model = MLPDynamicsModel(
    'dynamics_model',
    real_env
)

model_env = TfEnv(normalize(ModelEnv(
    env=HalfCheetahEnv(),
    dynamics_model=dynamics_model,
)))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=real_env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=real_env.spec)

algo = VINETRPO(
    real_env=real_env,
    model_env=model_env,
    policy=policy,
    baseline=baseline,
    dynamics_model=dynamics_model,
    num_paths=1,
    n_itr=40,
    discount=0.99,
    step_size=0.01,

)

algo.train()