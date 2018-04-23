


from nose2.tools import such
from rllab_maml.envs.box2d.cartpole_env import CartpoleEnv
from rllab_maml.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab_maml.algos.trpo import TRPO
from rllab_maml.baselines.zero_baseline import ZeroBaseline

with such.A("Issue #3") as it:
    @it.should("be fixed")
    def test_issue_3():
        """
        As reported in https://github.com/rllab_maml/rllab_maml/issues/3, the adaptive_std parameter was not functioning properly
        """
        env = CartpoleEnv()
        policy = GaussianMLPPolicy(
            env_spec=env,
            adaptive_std=True
        )
        baseline = ZeroBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=100,
            n_itr=1
        )
        algo.train()

it.createTests(globals())
