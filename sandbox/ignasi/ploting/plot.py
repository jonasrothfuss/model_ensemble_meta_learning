import numpy as np
import matplotlib.pyplot as plt
import csv
import os.path as osp
import os
import json
import argparse



def plot_exp(parser):
    # prefix = '/home/ignasi/GitRepos/rllab-thanard/data/s3/'
    # env = 'half-cheetah'
    # env = 'humanoid'
    # path = osp.join(prefix, env)
    path = parser.path
    if path[0] != '/':
        path = osp.join('/home/ignasi/GitRepos/model_ensemble_meta_learning/', path)
    experiments = os.listdir(path)
    for exp in experiments:
            try:
                import pdb; pdb.set_trace()
                header = next(csv.reader(open(osp.join(path, exp, 'progress.csv'), 'r')))
                id_return = header.index('AverageReturn')
                _data = np.genfromtxt(osp.join(path, exp, 'progress.csv'), delimiter=',', skip_header=True, dtype=np.float32).T
                data = _data[id_return]
                plt.plot(data)
            except Exception as e:
                pass
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, help='specify absolute path')
    parser.add_argument('--save_fig', '-s', type=bool, default=False,
                        help='Save the figure')

    args = parser.parse_args()
    plot_exp(args)
