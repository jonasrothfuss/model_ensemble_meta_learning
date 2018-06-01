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

from sandbox.jonas.envs.own_envs import PointEnvMAML
from sandbox.jonas.envs.mujoco import AntEnvRandParams, HalfCheetahEnvRandParams, HopperEnvRandParams
from sandbox.jonas.envs.mujoco import Reacher5DofEnvRandParams
from sandbox.jonas.envs.mujoco.cheetah_env import HalfCheetahEnv


import tensorflow as tf
import sys
import argparse
import random

EXP_PREFIX = 'model-ensemble-maml-hyperparam-search'

ec2_instance = 'm4.2xlarge'



def run_train_task(vv):

    env = TfEnv(normalize(vv['env'](log_scale_limit=vv['log_scale_limit'])))

    dynamics_model = MLPDynamicsEnsemble(
        name="dyn_model",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes_model'],
        weight_normalization=vv['weight_normalization_model'],
        num_models=vv['num_models'],
        optimizer=vv['optimizer_model']
    )

    policy = MAMLImprovedGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=vv['hidden_sizes_policy'],
        hidden_nonlinearity=vv['hidden_nonlinearity_policy'],
        grad_step_size=vv['fast_lr'],
        trainable_step_size=vv['trainable_step_size'],
        bias_transform=vv['bias_transform']
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = ModelMAMLTRPO(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        baseline=baseline,
        n_itr=vv['n_itr'],
        n_iter=vv['n_itr'],
        batch_size_env_samples=vv['batch_size_env_samples'],
        batch_size_dynamics_samples=vv['batch_size_dynamics_samples'],
        initial_random_samples=vv['initial_random_samples'],
        dynamic_model_epochs=vv['dynamic_model_epochs'],
        num_maml_steps_per_iter=vv['num_maml_steps_per_iter'],
        max_path_length=vv['path_length'],
        discount=vv['discount'],
        step_size=vv["meta_step_size"],
        num_grad_updates=1,
        retrain_model_when_reward_decreases=vv['retrain_model_when_reward_decreases'],
        reset_policy_std=vv['reset_policy_std'],
        reinit_model_cycle=vv['reinit_model_cycle'],
        frac_gpu=vv.get('frac_gpu', 1),
        vine_max_path_length=vv['vine_max_path_length'],
        n_vine_branch=vv['n_vine_branch'],
        n_vine_init_obs=vv['n_vine_init_obs'],
        noise_init_obs=vv['noise_init_obs'],
        log_real_data=True,
        # optimizer_args={'cg_iters': 15}

    )
    algo.train()

def run_experiment(argv):

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
    vg.add('env', ['AntEnvRandParams'])
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
    vg.add('vine_max_path_length', [30, 50])
    vg.add('n_vine_branch', [3, 5])
    vg.add('n_vine_init_obs', [5000])
    vg.add('noise_init_obs', [0])

    variants = vg.variants()

    default_dict =dict(exp_prefix=EXP_PREFIX,
            snapshot_mode="all",
            periodic_sync=True,
            sync_s3_pkl=True,
            sync_s3_log=True,
            python_command="python3",
            pre_commands=["yes | pip install tensorflow=='1.6.0'",
                          "pip list",
                          "yes | pip install --upgrade cloudpickle"],
            use_cloudpickle=True,
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
            n_parallel = int(info["vCPU"])
            use_gpu = False
        else:
            n_parallel = 12
            use_gpu = False

        if args.mode == 'ec2':


            config.AWS_INSTANCE_TYPE = ec2_instance
            config.AWS_SPOT_PRICE = str(info["price"])


            print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(variants)))
            print('Running on type {}, with price {}, on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                               config.AWS_SPOT_PRICE, ), str(subnets))

        # ----------------------- TRAINING ---------------------------------------
        exp_ids = random.sample(range(1, 1000), len(variants))
        subnets = cheapest_subnets(ec2_instance, num_subnets=3)
        for v, exp_id in zip(variants, exp_ids):
            exp_name = "%s_vine_mp%i_vine_b%i_vine_init_obs%i_seed%i_id_%i" % (v['env'], v['vine_max_path_length'],
                                                                               v['n_vine_branch'], v['n_vine_init_obs'],
                                                                               v['seed'], exp_id)
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
                use_gpu=use_gpu,
                # Number of parallel workers for sampling
                n_parallel=n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="all",
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