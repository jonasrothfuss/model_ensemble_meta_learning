import os

os.environ['THEANO_FLAGS'] = 'mode=FAST_COMPILE,optimizer=None'

from rllab_maml.algos.vpg import VPG
from rllab_maml.envs.box2d.cartpole_env import CartpoleEnv
from rllab_maml.baselines.zero_baseline import ZeroBaseline
from rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab_maml.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab_maml.policies.gaussian_mlp_policy import GaussianMLPPolicy
from nose2 import tools


baselines = [ZeroBaseline, LinearFeatureBaseline, GaussianMLPBaseline]


@tools.params(*baselines)
def test_baseline(baseline_cls):
    env = CartpoleEnv()
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(6,))
    baseline = baseline_cls(env_spec=env.spec)
    algo = VPG(
        env=env, policy=policy, baseline=baseline,
        n_itr=1, batch_size=1000, max_path_length=100
    )
    algo.train()
