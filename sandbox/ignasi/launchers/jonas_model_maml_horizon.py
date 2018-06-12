from rllab.misc.instrument import VariantGenerator
from rllab import config
from rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab_maml.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.jonas.envs.normalized_env import normalize
from sandbox.jonas.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.jonas.policies.maml_improved_gauss_mlp_policy import MAMLImprovedGaussianMLPPolicy
from sandbox.jonas.dynamics.dynamics_ensemble import MLPDynamicsEnsemble
from sandbox.ignasi.dynamics.probabilistic_dynamics_ensemble import MLPProbabilisticDynamicsEnsemble
from sandbox.jonas.algos.ModelMAML.model_maml_trpo import ModelMAMLTRPO
from experiments.helpers.ec2_helpers import cheapest_subnets
from experiments.helpers.run_multi_gpu import run_multi_gpu

from sandbox.jonas.envs.own_envs import PointEnvMAML
from sandbox.jonas.envs.mujoco import AntEnvRandParams, HalfCheetahEnvRandParams, HopperEnvRandParams, WalkerEnvRandomParams,\
    SwimmerEnvRandParams, HumanoidEnvRandParams, PR2EnvRandParams
from sandbox.jonas.envs.mujoco import Reacher5DofEnvRandParams


import tensorflow as tf
import sys
import argparse
import random
import os

EXP_PREFIX = 'model-ensemble-maml-good-paper-big-buffer'

ec2_instance = 'm4.xlarge'
NUM_EC2_SUBNETS = 3


def run_train_task(vv):
    env = TfEnv(normalize(vv['env'](log_scale_limit=vv['log_scale_limit'])))

    dynamics_model = vv['dynamics_model'](
        name="dyn_model",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes_model'],
        weight_normalization=vv['weight_normalization_model'],
        num_models=vv['num_models'],
        optimizer=vv['optimizer_model'],
    )

    policy = MAMLImprovedGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes_policy'],
        hidden_nonlinearity=vv['hidden_nonlinearity_policy'],
        grad_step_size=vv['fast_lr'],
        trainable_step_size=vv['trainable_step_size'],
        bias_transform=vv['bias_transform'],
        param_noise_std=vv['param_noise_std']
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = ModelMAMLTRPO(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        baseline=baseline,
        n_itr=vv['n_itr'],
        batch_size_env_samples=vv['batch_size_env_samples'],
        batch_size_dynamics_samples=vv['batch_size_dynamics_samples'],
        meta_batch_size=vv['meta_batch_size'],
        initial_random_samples=vv['initial_random_samples'],
        dynamic_model_max_epochs=vv['dynamic_model_epochs'],
        num_maml_steps_per_iter=vv['num_maml_steps_per_iter'],
        reset_from_env_traj=vv['reset_from_env_traj'],
        max_path_length_env=vv['path_length_env'],
        max_path_length_dyn=vv['path_length_dyn'],
        discount=vv['discount'],
        step_size=vv["meta_step_size"],
        num_grad_updates=1,
        retrain_model_when_reward_decreases=vv['retrain_model_when_reward_decreases'],
        reset_policy_std=vv['reset_policy_std'],
        reinit_model_cycle=vv['reinit_model_cycle'],
        frac_gpu=vv.get('frac_gpu', 0.85),
        log_real_performance=True,
        clip_obs=vv['clip_obs']
    )
    algo.train()

def run_experiment(argv):

    # -------------------- Parse Arguments -----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')
    parser.add_argument('--n_gpu', type=int, default=0,
                        help='Number of GPUs')
    parser.add_argument('--ctx', type=int, default=4,
                        help='Number of tasks per GPU')

    args = parser.parse_args(argv[1:])

    # -------------------- Define Variants -----------------------------------
    vg = VariantGenerator()

    vg.add('seed', [10, 20, 30])
    # env spec
    vg.add('env', ['HalfCheetahEnvRandParams', 'HopperEnvRandParams', 'WalkerEnvRandomParams', 'PR2EnvRandParams'])
    vg.add('log_scale_limit', [0.0])
    vg.add('path_length_env', [200])

    # Model-based MAML algo spec
    vg.add('path_length_dyn', [None])
    vg.add('n_itr', [100])
    vg.add('fast_lr', [0.001, 0.005])
    vg.add('meta_step_size', [0.01])
    vg.add('meta_batch_size', [20])  # must be a multiple of num_models
    vg.add('discount', [0.99])
    vg.add('batch_size_env_samples', [2])
    vg.add('batch_size_dynamics_samples', [50])
    vg.add('initial_random_samples', [None])
    vg.add('dynamic_model_epochs', [(200, 200)])
    vg.add('num_maml_steps_per_iter', [30])
    vg.add('retrain_model_when_reward_decreases', [False])
    vg.add('reset_from_env_traj', [False])
    vg.add('num_models', [5])
    vg.add('trainable_step_size', [False])

    # neural network configuration
    vg.add('hidden_nonlinearity_policy', ['tanh'])
    vg.add('hidden_nonlinearity_model', ['relu'])
    vg.add('hidden_sizes_policy', [(32, 32)])
    vg.add('hidden_sizes_model', [(512, 512)])
    vg.add('weight_normalization_model', [True])
    vg.add('reset_policy_std', [False])
    vg.add('reinit_model_cycle', [0])
    vg.add('optimizer_model', ['adam'])
    vg.add('policy', ['MAMLImprovedGaussianMLPPolicy'])
    vg.add('dynamics_model', ['MLPDynamicsEnsemble'])
    vg.add('bias_transform', [False])
    vg.add('param_noise_std', [0.0])
    vg.add('clip_obs', [True])
    # vg.add('nm_mbs_envs', [(5, 10, 2), (10, 10, 2), (10, 20, 1), (20, 20, 1)])

    # other stuff
    vg.add('exp_prefix', [EXP_PREFIX])


    variants = vg.variants()

    default_dict = dict(exp_prefix=EXP_PREFIX,
                        snapshot_mode="gap",
                        snapshot_gap=5,
                        periodic_sync=True,
                        sync_s3_pkl=True,
                        sync_s3_log=True,
                        python_command="python3",
                        pre_commands=["yes | pip install tensorflow=='1.6.0'",
                                      "pip list",
                                      "yes | pip install --upgrade cloudpickle"],
                        use_cloudpickle=True,
                        variants=variants)

    if args.mode == 'mgpu':
        current_path = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_path, 'jonas_run_gpu_model_ensemble_maml_trpo_train.py')
        n_gpu = args.n_gpu
        if n_gpu == 0:
            n_gpu = len(os.listdir('/proc/driver/nvidia/gpus'))
        run_multi_gpu(script_path, default_dict, n_gpu=n_gpu, ctx_per_gpu=args.ctx)

    else:
        # ----------------------- AWS conficuration ---------------------------------
        if args.mode == 'ec2':
            info = config.INSTANCE_TYPE_INFO[ec2_instance]
            n_parallel = int(info["vCPU"])
        else:
            n_parallel = 12

        if args.mode == 'ec2':


            config.AWS_INSTANCE_TYPE = ec2_instance
            config.AWS_SPOT_PRICE = str(info["price"])
            subnets = [ 'us-west-1a', 'us-west-1b', 'us-west-1c']#cheapest_subnets(ec2_instance, num_subnets=NUM_EC2_SUBNETS)
            print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(variants)))
            print('Running on type {}, with price {}, on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                               config.AWS_SPOT_PRICE, ), str(subnets))

        # ----------------------- TRAINING ---------------------------------------
        exp_ids = random.sample(range(1, 1000), len(variants))
        for v, exp_id in zip(variants, exp_ids):
            exp_name = "model_ensemble_maml_train_env_%s_%i_%i_%i_%i_id_%i" % (v['env'], v['path_length_env'], v['num_maml_steps_per_iter'],
                                                           v['batch_size_env_samples'], v['seed'], exp_id)
            v = instantiate_class_stings(v)

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


            run_experiment_lite(
                run_train_task,
                exp_prefix=EXP_PREFIX,
                exp_name=exp_name,
                # Number of parallel workers for sampling
                n_parallel=n_parallel,
                snapshot_mode="gap",
                snapshot_gap=5,
                periodic_sync=True,
                sync_s3_pkl=True,
                sync_s3_log=True,
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                seed=v["seed"],
                python_command="python3",
                pre_commands=["yes | pip install tensorflow=='1.6.0'",
                              "pip list",
                              "yes | pip install --upgrade cloudpickle"],
                mode=args.mode,
                use_cloudpickle=True,
                variant=v,
            )


def instantiate_class_stings(v):
    v['env'] = globals()[v['env']]
    v['dynamics_model'] = globals()[v['dynamics_model']]

    if 'nm_mbs_envs' in v.keys():
        v['num_models'] = v['nm_mbs_envs'][0]
        v['meta_batch_size'] = v['nm_mbs_envs'][1]
        v['batch_size_env_samples'] = v['nm_mbs_envs'][2]

    # optimizer
    if v['optimizer_model'] == 'sgd':
        v['optimizer_model'] = tf.train.GradientDescentOptimizer
    elif v['optimizer_model'] == 'adam':
        v['optimizer_model'] = tf.train.AdamOptimizer
    elif v['optimizer_model'] == 'momentum':
        v['optimizer_model'] = tf.train.MomentumOptimizer

    # nonlinearlity
    for nonlinearity_key in ['hidden_nonlinearity_policy', 'hidden_nonlinearity_model']:
        if v[nonlinearity_key] == 'relu':
            v[nonlinearity_key] = tf.nn.relu
        elif v[nonlinearity_key] == 'tanh':
            v[nonlinearity_key] = tf.tanh
        elif v[nonlinearity_key] == 'elu':
            v[nonlinearity_key] = tf.nn.elu
        else:
            raise NotImplementedError('Not able to recognize spicified hidden_nonlinearity: %s' % v['hidden_nonlinearity'])
    return v


if __name__ == "__main__":
    run_experiment(sys.argv)
