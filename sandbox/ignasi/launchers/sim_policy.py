import argparse
import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=25,
                        help='Speedup')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        if 'real_env' not in data.keys():
            env = data['env']
        else:
            env = data['real_env']
        while True:
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, speedup=args.speedup, always_return_paths=True)
            print('Return:  ', path['rewards'].sum())
            if not query_yes_no('Continue simulation?'):
                break