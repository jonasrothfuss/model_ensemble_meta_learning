from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import lasagne.nonlinearities as NL
import sys
from sandbox.jonas.envs.mujoco import HalfCheetahEnvRandParams, HopperEnvRandParams, AntEnvRandParams

from rllab.misc.instrument import VariantGenerator, variant
from rllab import config
import argparse
import random
import os

from sandbox.jonas.envs.mujoco import AntEnvRandParams, HalfCheetahEnvRandParams, HopperEnvRandParams, \
    WalkerEnvRandomParams, SwimmerEnvRandParams, PR2EnvRandParams
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from baselines import bench, logger
import multiprocessing
import rllab

import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import time
import tensorflow as tf
from mpi4py import MPI


EXP_PREFIX = 'ddpg-baselines'

ec2_instance = 'c4.2xlarge'

# configure instance

subnets = [
    'us-east-2a',
    'us-east-2b',
    'us-east-2c',
]

info = config.INSTANCE_TYPE_INFO[ec2_instance]
config.AWS_INSTANCE_TYPE = ec2_instance
config.AWS_SPOT_PRICE = str(info["price"])


def run_train_task(vv):


    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    log_dir = os.path.join(rllab.config.LOG_DIR, 'local', vv['exp_prefix'], vv['exp_name'])
    logger.configure(dir=log_dir)

    # Create envs.
    env = HalfCheetahEnvRandParams(log_scale_limit=0.0, max_path_length=vv['path_length'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))


    eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in vv['noise_type'].split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                        sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=vv['layer_norm'])
    actor = Actor(nb_actions, layer_norm=vv['layer_norm'])

    # Seed everything to make things reproducible.
    seed = vv['seed'] + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env,
                   eval_env=eval_env,
                   param_noise=param_noise,
                   action_noise=action_noise,
                   actor=actor,
                   critic=critic,
                   memory=memory,
                   actor_lr=vv['actor_lr'],
                   batch_size=vv['batch_size'],
                   clip_norm=vv['clip_norm'],
                   critic_l2_reg=vv['critic_l2_reg'],
                   critic_lr=vv['critic_lr'],
                   gamma=vv['discount'],
                   nb_epoch_cycles=vv['nb_epoch_cycles'],
                   nb_epochs=vv['nb_epochs'],
                   nb_eval_steps=vv['nb_eval_steps'],
                   nb_rollout_steps=vv['path_length'],
                   nb_train_steps=vv['nb_train_steps'],
                   normalize_observations=vv['normalize_observations'],
                   normalize_returns=vv['normalize_returns'],
                   popart=vv['popart'],
                   render=vv['render'],
                   render_eval=vv['render_eval'],
                   reward_scale=vv['reward_scale']
                   )

    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def run_experiment(argv):

    # -------------------- Parse Arguments -----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')

    args = parser.parse_args(argv[1:])

    # -------------------- Define Variants -----------------------------------

    vg = VariantGenerator()
    vg.add('env', ['HalfCheetahEnvRandParams', 'AntEnvRandParams', 'WalkerEnvRandomParams',
                   'SwimmerEnvRandParams', 'HopperEnvRandParams', 'PR2EnvRandParams'])
    vg.add('total_timesteps', [int(10**8)])
    vg.add('seed', [31, 41, 32])
    vg.add('discount', [0.99])
    vg.add('path_length', [200])
    vg.add('hidden_nonlinearity', ['tanh'])
    vg.add('hidden_sizes', [(32, 32)])
    vg.add('noise_type', ['adaptive-param_0.2'])
    vg.add('layer_norm', [True])
    vg.add('actor_lr', [0.00001])
    vg.add('batch_size', [64])
    vg.add('clip_norm', [None])
    vg.add('critic_l2_reg', [0.01])
    vg.add('critic_lr', [0.001])
    vg.add('nb_epoch_cycles', [25])
    vg.add('nb_epochs', [2000])
    vg.add('nb_eval_steps', [100])
    vg.add('nb_train_steps', [50])
    vg.add('normalize_observations', [True])
    vg.add('normalize_returns', [False])
    vg.add('popart', [False])
    vg.add('render', [False])
    vg.add('render_eval', [False])
    vg.add('reward_scale', [1.0])



    variants = vg.variants()
    from pprint import pprint
    pprint(variants)

    # ----------------------- AWS conficuration ---------------------------------
    if args.mode == 'ec2':
        n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
    else:
        n_parallel = 6

    if args.mode == 'ecs':
        print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(variants)))
        print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                       config.AWS_SPOT_PRICE,
                                                                                       n_parallel), *subnets)

    # ----------------------- TRAINING ---------------------------------------
    exp_ids = random.sample(range(1, 1000), len(variants))
    for v, exp_id in zip(variants, exp_ids):
        exp_name = "ddpg_%s_%i_%i_id_%i" % (v['env'], v['path_length'], v['seed'], exp_id)

        v['exp_name'] = exp_name
        v['exp_prefix'] = EXP_PREFIX

        v = instantiate_class_stings(v)

        subnet = random.choice(subnets)
        config.AWS_REGION_NAME = subnet[:-1]
        config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
            config.AWS_REGION_NAME]
        config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
            config.AWS_REGION_NAME]
        config.AWS_SECURITY_GROUP_IDS = \
            config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
                config.AWS_REGION_NAME]

        run_experiment_lite(
            run_train_task,
            exp_prefix=EXP_PREFIX,
            exp_name=exp_name,
            # Number of parallel workers for sampling
            n_parallel=n_parallel,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            sync_s3_pkl=True,
            periodic_sync=True,
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=v["seed"],
            #sync_all_data_node_to_s3=True,
            python_command="python3", #sys.executable,
            pre_commands=["yes | pip install tensorflow=='1.6.0'",
                          "yes | pip install --upgrade cloudpickle",
                          "yes | pip install gym==0.10.5"],
            mode=args.mode,
            use_cloudpickle=True,
            variant=v,
        )



def instantiate_class_stings(v):
    v['env'] = globals()[v['env']]

    if v['hidden_nonlinearity'] == 'relu':
        v['hidden_nonlinearity'] = NL.rectify
    elif v['hidden_nonlinearity'] == 'tanh':
        v['hidden_nonlinearity'] = NL.tanh
    elif v['hidden_nonlinearity'] == 'elu':
        v['hidden_nonlinearity'] = NL.elu
    else:
        raise NotImplementedError('Not able to recognize spicified hidden_nonlinearity: %s' % v['hidden_nonlinearity'])
    return v

if __name__ == "__main__":
    run_experiment(sys.argv)