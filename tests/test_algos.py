import os

from rllab_maml.algos.cem import CEM
from rllab_maml.algos.cma_es import CMAES
from rllab_maml.algos.erwr import ERWR

os.environ['THEANO_FLAGS'] = 'device=cpu,mode=FAST_COMPILE,optimizer=None'

from rllab_maml.algos.vpg import VPG
from rllab_maml.algos.tnpg import TNPG
from rllab_maml.algos.ppo import PPO
from rllab_maml.algos.trpo import TRPO
from rllab_maml.algos.reps import REPS
from rllab_maml.algos.ddpg import DDPG
from rllab_maml.envs.grid_world_env import GridWorldEnv
from rllab_maml.envs.box2d.cartpole_env import CartpoleEnv
from rllab_maml.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab_maml.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab_maml.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab_maml.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab_maml.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab_maml.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab_maml.exploration_strategies.ou_strategy import OUStrategy
from rllab_maml.baselines.zero_baseline import ZeroBaseline
from nose2 import tools
import numpy as np

common_batch_algo_args = dict(
    n_itr=1,
    batch_size=1000,
    max_path_length=100,
)

algo_args = {
    VPG: common_batch_algo_args,
    TNPG: dict(common_batch_algo_args,
        optimizer_args=dict(
            cg_iters=1,
        ),
    ),
    TRPO: dict(common_batch_algo_args,
        optimizer_args=dict(
            cg_iters=1,
        ),
    ),
    PPO: dict(common_batch_algo_args,
        optimizer_args=dict(
            max_penalty_itr=1,
            max_opt_itr=1
        ),
    ),
    REPS: dict(common_batch_algo_args,
        max_opt_itr=1,
    ),
    DDPG: dict(
        n_epochs=1,
        epoch_length=100,
        batch_size=32,
        min_pool_size=50,
        replay_pool_size=1000,
        eval_samples=100,
    ),
    CEM: dict(
        n_itr=1,
        max_path_length=100,
        n_samples=5,
    ),
    CMAES: dict(
        n_itr=1,
        max_path_length=100,
        batch_size=1000,
    ),
    ERWR: common_batch_algo_args,
}

polopt_cases = []
for algo in [VPG, TNPG, PPO, TRPO, CEM, CMAES, ERWR, REPS]:
    polopt_cases.extend([
        (algo, GridWorldEnv, CategoricalMLPPolicy),
        (algo, CartpoleEnv, GaussianMLPPolicy),
        (algo, GridWorldEnv, CategoricalGRUPolicy),
        (algo, CartpoleEnv, GaussianGRUPolicy),
    ])


@tools.params(*polopt_cases)
def test_polopt_algo(algo_cls, env_cls, policy_cls):
    print("Testing %s, %s, %s" % (algo_cls.__name__, env_cls.__name__, policy_cls.__name__))
    env = env_cls()
    policy = policy_cls(env_spec=env.spec, )
    baseline = ZeroBaseline(env_spec=env.spec)
    algo = algo_cls(env=env, policy=policy, baseline=baseline, **(algo_args.get(algo_cls, dict())))
    algo.train()
    assert not np.any(np.isnan(policy.get_param_values()))


def test_ddpg():
    env = CartpoleEnv()
    policy = DeterministicMLPPolicy(env.spec)
    qf = ContinuousMLPQFunction(env.spec)
    es = OUStrategy(env.spec)
    algo = DDPG(
        env=env, policy=policy, qf=qf, es=es,
        n_epochs=1,
        epoch_length=100,
        batch_size=32,
        min_pool_size=50,
        replay_pool_size=1000,
        eval_samples=100,
    )
    algo.train()
