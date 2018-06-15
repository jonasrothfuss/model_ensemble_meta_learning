import numpy as np
from rllab.misc.ext import flatten
from pprint import pprint
from collections import OrderedDict, defaultdict

def filter(exps_data, filters={}):
    print("before filtering", len(exps_data), 'exps')
    keep_array = []
    if filters:
        for i, exp in enumerate(exps_data):
            keep_array.append(all([((filter_key in exp['flat_params']) and (exp['flat_params'][filter_key] == filter_val))
                                              for filter_key, filter_val in filters.items()]))
        exps_data_filtered = np.array(exps_data)
        exps_data_filtered = exps_data_filtered[keep_array]
    else:
        exps_data_filtered = exps_data
    print("after filtering", len(exps_data_filtered), 'exps')
    return exps_data_filtered

def group_by(exp_data, group_by_key=None):
    split_dict = OrderedDict()
    for exp in exp_data:
        if group_by_key == 'exp_name':
            exp['flat_params']['exp_name'] = exp['flat_params']['exp_name'].replace('-', '_')
            key_str = str(exp['flat_params'][group_by_key]).split('_')[2]
            if key_str == 'train':
                key_str = 'me-trpo'
            elif key_str == 'maml':
                key_str = 'ours'
        elif group_by_key == 'env.$class':
            key_str = str(exp['flat_params'][group_by_key]).split('.')[-1]
            if key_str[-13:] == 'EnvRandParams':
                key_str = key_str[:-13]
            elif key_str[-15:] == 'EnvRandomParams':
                key_str = key_str[:-15] + '2D'
            else:
                key_str = key_str[:-3]
        else:
            key_str = str(exp['flat_params'][group_by_key])
        if key_str in split_dict.keys():
            split_dict[key_str].append(exp)
        else:
            split_dict[key_str] = [exp]
    return split_dict

def prepare_data_for_plot(exp_data, x_key='n_timesteps', y_key=None, sup_y_key=None, round_x=None):
    x_y_tuples = []
    for exp in exp_data:
        if sup_y_key is not None and sup_y_key in exp['progress'].keys():
            x_y_tuples.extend(list(zip(exp['progress'][x_key], exp['progress'][sup_y_key])))
        else:
            x_y_tuples.extend(list(zip(exp['progress'][x_key], exp['progress'][y_key])))
    x_y_dict = defaultdict(list)
    for k, v in x_y_tuples:
        if round_x is not None:
            x_y_dict[(k//round_x) * round_x].append(v)
        else:
            x_y_dict[k].append(v)
    means, stddevs = [], []
    for key in sorted(x_y_dict.keys()):
        means.append(np.mean(x_y_dict[key]))
        stddevs.append(np.std(x_y_dict[key]))
    return np.array(sorted(x_y_dict.keys())), np.array(means), np.array(stddevs)