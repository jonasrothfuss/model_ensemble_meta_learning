import argparse
import tensorflow as tf
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.gym_env import GymEnv
import rllab.config as config
import os.path

get_eval_data_path=dict(
    reacher="data_upload/policy_validation_reset_inits_reacher.save",
    swimmer="data_upload/policy_validation_inits_swimmer.save",
    snake="data_upload/policy_validation_reset_inits_snake.save",
    half_cheetah="data_upload/policy_validation_reset_inits_half_cheetah.save"
)

from private_examples.com_swimmer_env import SwimmerEnv
from private_examples.com_snake_env import SnakeEnv
from private_examples.reacher_env import ReacherEnv, gym_to_local
from private_examples.com_half_cheetah_env import HalfCheetahEnv
from private_examples.com_hopper_env import HopperEnv
from private_examples.com_walker_env import WalkerEnv
from private_examples.com_ant_env import AntEnv
from private_examples.com_simple_humanoid_env import SimpleHumanoidEnv
from private_examples.com_humanoid_env import HumanoidEnv

def get_env(env_name):
    if env_name == 'reacher':
        env = TfEnv(GymEnv("Reacher-v1", record_video=False, record_log=False))
        gym_to_local()
        env.wrapped_env.env.env = ReacherEnv()
        return env
    elif env_name == 'snake':
        return TfEnv(normalize(SnakeEnv()))
    elif env_name == 'swimmer':
        return TfEnv(normalize(SwimmerEnv()))
    elif env_name == 'half_cheetah':
        return TfEnv(normalize(HalfCheetahEnv()))
    elif env_name == 'hopper':
        return TfEnv(normalize(HopperEnv()))
    elif env_name == 'walker':
        # return TfEnv(GymEnv('Walker2d-v1',
        #                     record_video=False,
        #                     record_log=False))
        return TfEnv(normalize(WalkerEnv()))
    elif env_name == 'ant':
        # return TfEnv(GymEnv('Ant-v1',
        #                     record_video=False,
        #                     record_log=False))
        return TfEnv(normalize(AntEnv()))
    elif env_name == 'humanoidstandup':
        return TfEnv(GymEnv('HumanoidStandup-v1',
                            record_video=False,
                            record_log=False))
    elif env_name == 'humanoid':
        # return TfEnv(GymEnv('Humanoid-v1',
        #                     record_video=False,
        #                     record_log=False))
        return TfEnv(normalize(HumanoidEnv()))
    elif env_name == 'simple_humanoid':
        return TfEnv(normalize(SimpleHumanoidEnv()))
    else:
        assert False, "Define the env from env_name."

def get_algo(env_name,
             use_eval,
             init_path,
             horizon,
             batch_size,
             n_itr):
    env = get_env(env_name)
    policy = GaussianMLPPolicy(
        name='policy',
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        # output_nonlinearity=tf.nn.tanh
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    kwargs = dict(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=horizon,
        n_itr=n_itr,
        discount=1.00,
        step_size=0.01,
    )
    if use_eval:
        kwargs["reset_init_path"] = os.path.join(config.PROJECT_PATH, get_eval_data_path[env_name])
        kwargs["horizon"] = horizon
    if init_path is not None:
        kwargs["initialized_path"]=init_path
    return TRPO(**kwargs)

def train(*_):
    algo = get_algo(options.env_name,
                    options.use_eval,
                    options.policy_init_path,
                    options.horizon,
                    options.batch_size,
                    options.niters)
    algo.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run trpo locally options')
    parser.add_argument('--env_name')
    parser.add_argument('-use_eval', action="store_true", default=False)
    parser.add_argument('--policy_init_path', default=None)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--niters', type=int, default=1000)
    parser.add_argument('-ec2', action="store_true", default=False)
    parser.add_argument('--nexps', type=int, default=1)
    parser.add_argument('--prefix', type=str, default=None)
    options = parser.parse_args()
    from sandbox.thanard.bootstrapping.run_model_based_rl import get_aws_config
    if options.prefix is None:
        prefix = '%s-mf-trpo' % options.env_name
    else:
        prefix = '%s-mf-trpo/%s' % (options.prefix, options.env_name)
    if options.ec2:
        for i in range(options.nexps):
            aws_config = get_aws_config(i, use_gpu=False)
            run_experiment_lite(
                train,
                exp_prefix=prefix,
                n_parallel=1,
                snapshot_mode='last',
                mode='ec2',
                aws_config=aws_config,
                variant=vars(options),
                seed=i
            )
            print(i)
    else:
        run_experiment_lite(
            train,
            exp_prefix=prefix,
            n_parallel=1,
            snapshot_mode='last',
            variant=vars(options),
            seed=1
        )