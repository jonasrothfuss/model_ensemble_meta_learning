import os
import os.path as osp
import joblib
from rllab import config
import numpy as np
import random
from gym.monitoring.video_recorder import ImageEncoder
import tensorflow as tf
import cv2
from rllab.misc import console
import argparse
import pdb
from rllab.misc.console import query_yes_no

frame_size = (500, 500)


def to_img(obs, frame_size=(100, 100)):
    return cv2.resize(np.cast['uint8'](obs), frame_size)
    # return cv2.resize(np.cast['uint8']((obs / 2 + 0.5) * 255.0), frame_size)
    # return obs


'''
folderpath
env
mode:
    real
    model
kwargs require
    [dynamics_in, dynamics_out, cost_np] or env
    [policy_in, policy_out] or policy
    sess (if the first case)

'''


def record(log_path,
           env,
           horizon,
           kwargs,
           modes=['real']):
    _sanity_check(kwargs, modes)
    init = env.reset()
    Os = {}
    total_costs = {}
    for mode in modes:
        Os[mode] = []
        total_costs[mode] = []
        output_path = osp.join(log_path, "%s.mp4" % mode)
        encoder = ImageEncoder(output_path=output_path,
                               frame_shape=frame_size + (3,),
                               frames_per_sec=60)
        print("Generating %s" % output_path)
        obs = init
        Os[mode].append(obs)
        inner_env = _get_inner_env(env)
        if mode == 'model':
            inner_env.reset(obs)
        image = inner_env.render(mode='rgb_array')
        total_cost = 0.0
        total_costs[mode].append(total_cost)
        for t in range(horizon):
            compressed_image = to_img(image, frame_size=frame_size)
            # cv2.imshow('frame{}'.format(t), compressed_image)
            # cv2.waitKey(10)
            encoder.capture_frame(compressed_image)
            action = _get_action(kwargs, obs)
            action = np.clip(action, *env.action_space.bounds)
            next_obs, reward, done, info = _step(kwargs, env, obs, action, mode)
            total_cost -= reward
            obs = next_obs
            Os[mode].append(obs)
            if mode == 'model':
                inner_env.reset(next_obs)
            image = inner_env.render(mode='rgb_array')
            # if done:
            #     break
            total_costs[mode].append(total_cost)
        print("%s cost: %f" % (mode, total_cost))
        encoder.close()
    if len(Os) == 2:
        _analyze_trajectories(Os, total_costs, log_path)

def _analyze_trajectories(Os, total_costs, log_path, dims_to_plot=None):
    model_traj = np.array(Os['model'])
    real_traj = np.array(Os['real'])
    state_error = np.square(model_traj - real_traj)
    horizon, n_dims = state_error.shape
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    if dims_to_plot is None:
        dims_to_plot = range(n_dims)
    for d in dims_to_plot:
        plt.plot(range(horizon), state_error[:, d], label='dim%d' % d)
    plt.xlabel('horizon')
    plt.ylabel('square error')
    plt.legend()
    plt.savefig(osp.join(log_path, 'state_error.png'))
    plt.close()

    for d in dims_to_plot:
        plt.plot(range(horizon), model_traj[:, d], label='model')
        plt.plot(range(horizon), real_traj[:, d], label='real')
        plt.xlabel('horizon')
        plt.ylabel('state %d' % d)
        plt.legend()
        plt.savefig(osp.join(log_path, 'dim%d.png' % d))
        plt.close()

    plt.plot(range(horizon), total_costs['model'], label='model')
    plt.plot(range(horizon), total_costs['real'], label='real')
    plt.xlabel('horizon')
    plt.ylabel('total cost')
    plt.legend()
    plt.savefig(osp.join(log_path, 'total_costs.png'))
    plt.close()
def _get_inner_env(env):
    # pdb.set_trace()
    if hasattr(env, 'wrapped_env'):
        return _get_inner_env(env.wrapped_env)
    elif hasattr(env, 'env'):
        return _get_inner_env(env.env)
    return env


def _sanity_check(kwargs, modes):
    # policy check
    if 'policy_in' in kwargs:
        assert 'policy_out' in kwargs
        assert 'sess' in kwargs
        assert 'policy' not in kwargs
    else:
        assert 'policy' in kwargs
    # dynamics check
    if 'model' in modes:
        assert 'dynamics_in' in kwargs
        assert 'dynamics_out' in kwargs
        assert 'cost_np' in kwargs
        assert 'sess' in kwargs


def _get_action(kwargs, obs):
    if 'policy_in' in kwargs:
        return kwargs['sess'].run(
            kwargs['policy_out'],
            feed_dict={
                kwargs['policy_in']: obs[None]
            })[0]
    return kwargs['policy'].get_action(obs)[1]["mean"]


def _step(kwargs, env, obs, action, mode):
    if mode == 'model':
        next_obs = kwargs['sess'].run(
            kwargs['dynamics_out'],
            feed_dict={
                kwargs['dynamics_in']: np.concatenate(
                    [obs, action]
                )[None]
            })[0]
        cost = kwargs['cost_np'](
            obs[None],
            action[None],
            next_obs[None]
        )
        return next_obs, -cost, False, None
    else:
        assert mode == 'real'
        return env.step(action)

if __name__ == '__main__':
    with tf.Session() as sess:
        '''
        Use this script to save videos of real and simulated trajectories.
        It also gives visualization of the prediction errors in each state
        dimension.
        Note that sim video only works when observation is the same as state,
        e.g., reacher, or when we have a reset function from observation, e.g.,
        swimmer.
        Example:
            python record_video.py "fullpath"
                --horizon 50
                --mode mr
                --policy ".../data_upload/paams-reacher.pkl"
        '''
        # TODO: add --model to choose scope and iteration of the model.
        # TODO: add --visualize choose state dimensions to visualize.
        parser = argparse.ArgumentParser()
        parser.add_argument('file', type=str,
                            help='path to the snapshot file', default='swimmer_normalized_com_cost')
        parser.add_argument('--horizon', type=int, default=100,
                            help='Max length of rollout')
        parser.add_argument('--mode',
                            type=str,
                            action='store',
                            default='r',
                            help='choose modes')
        parser.add_argument('--ckpt',
                            type=str,
                            default='training_logs/policy-and-models-final.ckpt')
        parser.add_argument('--policy',
                            type=str,
                            default='')
        args = parser.parse_args()
        modes = []
        if 'r' in args.mode:
            modes.append('real')
        if 'm' in args.mode:
            modes.append('model')

        np.random.seed(0)
        random.seed(0)
        filepath = args.file
        pkl_file = osp.join(filepath,
                            "params.pkl")
        if osp.isfile(args.policy):
            pkl_file = args.policy

        # Load params.pkl
        data = joblib.load(pkl_file)
        policy = data["policy"]
        env = data['env']
        # Comment the lines below out.
        # from private_examples.reacher_env import ReacherEnv
        #
        # env.wrapped_env.env.env = ReacherEnv()
        # env = SwimmerEnv()
        kwargs = {'policy': policy}

        # Load tf session ckpt
        if "model" in modes:
            ckpt_path = osp.join(filepath,
                                 args.ckpt)
            meta_path = ckpt_path + '.meta'
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, ckpt_path)
            if osp.exists(meta_path):
                kwargs = {
                    'sess': sess,
                    'policy': policy,
                    # 'policy_in': tf.get_collection('policy_in')[0],
                    # 'policy_out': tf.get_collection('policy_out')[0],
                    'dynamics_in': tf.get_collection('dynamics_in')[0],
                    'dynamics_out': tf.get_collection('training_dynamics_out')[0],
                    'cost_np': _get_inner_env(env).cost_np
                }
        count = 0
        while True:
            while True:
                log_path = osp.join(filepath, 'traj-%d' % count)
                if not osp.exists(log_path):
                    break
                count += 1
            os.makedirs(log_path)
            record(log_path,
                   env,
                   args.horizon,
                   kwargs,
                   modes)
            if not query_yes_no('Continue simulation?'):
                break