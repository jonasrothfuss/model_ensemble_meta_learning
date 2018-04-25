from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import run_experiment_lite
import rllab.config as config
from rllab.envs.proxy_env import ProxyEnv

import tensorflow as tf
import joblib
import numpy as np
import glob
import os
import json
import copy
import sys
import glob

def prepare_evaluation_runs(exp_prefix_dir, num_sampled_envs=5):
    """
    Given a directory of train runs, the method loads the provided data and creates
    variant configurations for evaluation runs
    :param exp_prefix_dir: directory with train run logging directories
    :param num_sampled_envs: number or environments with samples parameters
    :return: eval_task_list - a list that contains ordered pairs like (eval_exp_name, eval_variant_dict)
    """

    assert os.path.isdir(exp_prefix_dir), "exp_prefix_dir must be directory"

    exp_dirs = glob.glob(os.path.join(exp_prefix_dir, '*/'))

    eval_task_list = [] #list that contains ordered pairs like (eval_exp_name, variant_dict)

    for exp_dir in exp_dirs:
        # get variant dir with additional params_pickle_file entry
        variant_dict = extract_files_from_dir(exp_dir)

        # generate num_sampled_envs seeds for sampling the environment params
        env_param_seeds = np.random.RandomState(variant_dict["seed"]).randint(0, 1000, size=(num_sampled_envs,))

        for env_seed in env_param_seeds:
            env_seed = int(env_seed)  # make sure that env seed is python integer to be JSON serializable

            # make copy of variant dict and add env_seed to it
            v = copy.deepcopy(variant_dict)
            v['env_param_seed'] = env_seed

            train_exp_name = os.path.dirname(exp_dir).split('/')[-1]
            if 'train' in train_exp_name:
                eval_exp_name = train_exp_name.replace('train', 'eval') + '_env_seed_%i'%env_seed
            else:
                eval_exp_name = train_exp_name + '_eval_env_seed_%i'%env_seed

            eval_task_list.append((eval_exp_name, v))

    assert len(eval_task_list) == len(exp_dirs) * num_sampled_envs
    return eval_task_list


def extract_files_from_dir(results_dir_path):
    """
    Checks if existent an then extracts relevant files (params.pkl, variant.json) from results_dir_path
    :param results_dir_path: directory which shall be evaluated
    :return: variant_dict with additional entries 'params_pickle_file'
    """
    assert os.path.isdir(results_dir_path)

    assert len(glob.glob(os.path.join(results_dir_path,'*params*.pkl'))) == 1, 'Directory must not contain more than one parameter file'
    params_pickle_file = glob.glob(os.path.join(results_dir_path,'*params*.pkl'))[0]

    assert len(glob.glob(os.path.join(results_dir_path,'*variant*.json'))) == 1, 'Directory must not contain more than one variant file'
    variant_json_path =  glob.glob(os.path.join(results_dir_path,'*variant*.json'))[0]
    with open(variant_json_path, 'r') as f:
        variant_dict = json.load(f)

    variant_dict["params_pickle_file"] = params_pickle_file

    return variant_dict

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

def load_saved_objects(variant_dict):
    """
    Loads policy, baseline and environment object from pickle which is specified in the variant_dict
    Warning: resets the tf graph

    :param variant_dict that must contain the params_pickle_file
    :return: loaded policy, baseline, environment a tensoflow session that is associated with the policy/baseline
    """
    tf.reset_default_graph()
    sess = tf.Session()
    sess.__enter__()
    loaded_data = joblib.load(variant_dict['params_pickle_file'])
    policy = loaded_data["policy"]
    baseline = loaded_data["baseline"]
    env = loaded_data["env"]
    return policy, baseline, env, sess

def get_local_exp_log_dir(exp_prefix, exp_name):
    ''' determines log path of experiment'''
    return os.path.join(config.LOG_DIR, 'local', exp_prefix, exp_name)

def get_env_class(env):
    while isinstance(env, ProxyEnv):
        env = env.wrapped_env
    return env.__class__