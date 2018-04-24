from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from experiments.envs.mujoco.half_cheetah_env_rand_param import HalfCheetahEnvRandParams
from rllab.misc.instrument import run_experiment_lite
import rllab.config as config

import tensorflow as tf
import joblib
import numpy as np
import glob
import os
import json
import copy
import sys

def evaluate_policy_transfer(eval_task_fun, results_dir_path, num_samped_envs=5):
    """
    evaluates a trained policy by performing further (fine-tuning) gradient steps on envs with sampled model parameters
    :param eval_task_fun: function that takes variant_dict as only arguments and runs evaluation experiments
    :param results_dir_path: path with params.pkl and variant.json file
    :param num_samped_envs: number or environments with samples environments
    """
    params_pickle_file, vv = extract_files_from_dir(results_dir_path)
    experiment_name = experiment_name_from_path(results_dir_path)
    experiment_prefix = experiment_prefix_from_path(results_dir_path)

    # create different seeds for sampling the env parameters
    env_param_seeds = np.random.RandomState(vv["seed"]).randint(0, 1000, size=(num_samped_envs,))

    # add infomatio to variant dict
    vv["params_pickle_file"] = params_pickle_file

    for env_seed in env_param_seeds:
        env_seed = int(env_seed) # make sure that env seed is python integer to be JSON serializable
        varian_dict = copy.deepcopy(vv)
        varian_dict['env_param_seed'] = env_seed

        run_experiment_lite(
            eval_task_fun,
            exp_prefix=experiment_prefix +"_eval",
            exp_name=experiment_name + "_eval_%i"%env_seed,
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=varian_dict["seed"],
            python_command=sys.executable,
            mode="local",
            use_cloudpickle=True,
            # mode="ec2",
            variant=varian_dict,
            # plot=True,
            # terminate_machine=False,
        )

def extract_files_from_dir(results_dir_path):
    """
    Checks whether the directory contains a params.pkl and variant.json file otherwise throws assertion error
    :param results_dir_path: directory which shall be evaluated
    :return: params_pickle_file,
    """
    assert os.path.isdir(results_dir_path)

    assert len(glob.glob(os.path.join(results_dir_path,'*params*.pkl'))) == 1, 'Directory must not contain more than one parameter file'
    params_pickle_file = glob.glob(os.path.join(results_dir_path,'*params*.pkl'))[0]

    assert len(glob.glob(os.path.join(results_dir_path,'*variant*.json'))) == 1, 'Directory must not contain more than one variant file'
    variant_json_path =  glob.glob(os.path.join(results_dir_path,'*variant*.json'))[0]
    with open(variant_json_path, 'r') as f:
        variant_dict = json.load(f)

    return params_pickle_file, variant_dict

def experiment_name_from_path(results_dir_path):
    return os.path.basename(results_dir_path)

def experiment_prefix_from_path(results_dir_path):
    return os.path.basename(os.path.dirname(results_dir_path))

def create_fixed_envs(env_class, num_sampled_envs, random_seed, **kwargs):
    env_seeds = np.random.RandomState(random_seed).randint(num_sampled_envs)
    envs = []
    for seed in env_seeds:
        env = TfEnv(normalize(env_class.__init__(random_seed=seed, **kwargs)))
        env.sample_and_fix_parameters()
        envs.append(env)
    return envs

def load_policy_and_baseline(variant_dict):
    """
    Loads policy and baseline object from pickle which is specified in the variant_dict
    Warning: resets the tf graph

    :param variant_dict that must contain the params_pickle_file
    :return: loaded policy, baseline and a tensoflow session that is associated with the policy/baseline
    """
    tf.reset_default_graph()
    sess = tf.Session()
    sess.__enter__()
    loaded_data = joblib.load(variant_dict['params_pickle_file'])
    policy = loaded_data["policy"]
    baseline = loaded_data["baseline"]
    return policy, baseline, sess

def get_local_exp_log_dir(exp_prefix, exp_name):
    ''' determines log path of experiment'''
    return os.path.join(config.LOG_DIR, 'local', exp_prefix, exp_name)