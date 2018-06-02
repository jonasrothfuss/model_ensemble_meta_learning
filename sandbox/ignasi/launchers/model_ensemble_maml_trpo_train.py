from rllab.misc.instrument import VariantGenerator
from rllab import config
from rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.jonas.envs.normalized_env import normalize
from sandbox.jonas.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.jonas.policies.maml_improved_gauss_mlp_policy import MAMLImprovedGaussianMLPPolicy
from sandbox.jonas.dynamics.dynamics_ensemble import MLPDynamicsEnsemble
from sandbox.ignasi.algos.ModelMAML.model_maml_trpo import ModelMAMLTRPO
from experiments.helpers.ec2_helpers import cheapest_subnets
from sandbox.ignasi.launchers.run_multi_gpu import run_multi_gpu
import os
from sandbox.ignasi.launchers.run_gpu_model_ensemble_maml_trpo_train import run_experiment

from sandbox.jonas.envs.own_envs import PointEnvMAML
from sandbox.jonas.envs.mujoco import AntEnvRandParams, HalfCheetahEnvRandParams, HopperEnvRandParams
from sandbox.jonas.envs.mujoco import Reacher5DofEnvRandParams
from sandbox.jonas.envs.mujoco.cheetah_env import HalfCheetahEnv
import datetime
import dateutil.tz

import tensorflow as tf
import sys
import argparse
import random

EXP_PREFIX = 'model-ensemble-maml-hyperparam-search'

ec2_instance = 'm4.2xlarge'


def run_script(argv):

    # -------------------- Parse Arguments -----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mgpu',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')
    parser.add_argument('-n_gpu', type=int, default=0)
    parser.add_argument('-ctx', type=int, default=2)

    args = parser.parse_args(argv[1:])

    # -------------------- Define Variants -----------------------------------

    vg = VariantGenerator()
    vg.add('env', ['HalfCheetahEnvRandParams'])
    vg.add('n_itr', [80])
    vg.add('log_scale_limit', [0.0])
    vg.add('fast_lr', [0.01])
    vg.add('meta_step_size', [0.01])
    vg.add('seed', [0, 10]) #TODO set back to [1, 11, 21, 31, 41]
    vg.add('discount', [0.99])
    vg.add('path_length', [100])
    vg.add('batch_size_env_samples', [10])
    vg.add('batch_size_dynamics_samples', [100])
    vg.add('initial_random_samples', [5000])
    vg.add('dynamic_model_epochs', [(100, 50)])
    vg.add('num_maml_steps_per_iter', [10])
    vg.add('hidden_nonlinearity_policy', ['tanh'])
    vg.add('hidden_nonlinearity_model', ['relu'])
    vg.add('hidden_sizes_policy', [(32, 32)])
    vg.add('hidden_sizes_model', [(1024, 1024)])
    vg.add('weight_normalization_model', [True])
    vg.add('reset_policy_std', [False])
    vg.add('reinit_model_cycle', [0])
    vg.add('optimizer_model', ['adam'])
    vg.add('retrain_model_when_reward_decreases', [False])
    vg.add('num_models', [10])
    vg.add('trainable_step_size', [False])
    vg.add('bias_transform', [False])
    vg.add('policy', ['MAMLImprovedGaussianMLPPolicy'])
    vg.add('vine_max_path_length', [50])
    vg.add('n_vine_branch', [5])
    vg.add('n_vine_init_obs', [5000])
    vg.add('noise_init_obs', [0])

    variants = vg.variants()

    default_dict = dict(exp_prefix=EXP_PREFIX,
            snapshot_mode="all",
            periodic_sync=True,
            sync_s3_pkl=True,
            sync_s3_log=True,
            python_command="python3",
            pre_commands=["yes | pip install tensorflow=='1.6.0'",
                          "pip list",
                          "yes | pip install --upgrade cloudpickle"],
            use_cloudpickle=True,
            use_gpu=True,
            n_parallel=12,
            mode='local',
            variants=variants)

    # ----------------------- AWS conficuration ---------------------------------
    if args.mode == 'mgpu':
        current_path = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_path, 'run_gpu_model_ensemble_maml_trpo_train.py')
        n_gpu = args.n_gpu
        if n_gpu == 0:
            n_gpu = len(os.listdir('/proc/driver/nvidia/gpus'))
        run_multi_gpu(script_path, default_dict, n_gpu=n_gpu, ctx_per_gpu=args.ctx)

    else:
        if args.mode == 'ec2':
            info = config.INSTANCE_TYPE_INFO[ec2_instance]
            default_dict['n_parallel'] = int(info["vCPU"])
            default_dict['use_gpu'] = False
            default_dict['mode'] = 'ec2'
            subnets = cheapest_subnets(ec2_instance, num_subnets=3)

            config.AWS_INSTANCE_TYPE = ec2_instance
            config.AWS_SPOT_PRICE = str(info["price"])


            print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(variants)))
            print('Running on type {}, with price {}, on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                               config.AWS_SPOT_PRICE, ), str(subnets))
        # ----------------------- TRAINING ---------------------------------------
        exp_dict = default_dict.copy()

        del exp_dict['variants']
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        for idx, v in enumerate(variants):
            if args.mode == 'ec2':
                subnet = random.choice(subnets)
                config.AWS_REGION_NAME = subnet[:-1]
                config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
                    config.AWS_REGION_NAME]
                config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
                    config.AWS_REGION_NAME]
                config.AWS_SECURITY_GROUP_IDS = \
                    config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
                        config.AWS_REGION_NAME]
            exp_dict['variant'] = v
            exp_dict['variant']['exp_name'] = "%s_%04d" % (timestamp, idx)
            run_experiment(exp_dict)


if __name__ == "__main__":
    run_script(sys.argv)
