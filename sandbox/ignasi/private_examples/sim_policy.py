import argparse

import joblib
import tensorflow as tf
import numpy as np
from rllab.misc import tensor_utils
import time
from rllab.misc.console import query_yes_no
np.set_printoptions(
    formatter={
        'float_kind': lambda x: "%.2f"%x
    }
)

'''
Rollout mlp policy deterministically - using mean.
'''
def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False, action_noise = 0.0):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a_sampled, agent_info = agent.get_action(o)
        # use the mean here for eval.
        a = agent_info['mean']
        a += np.random.randn(len(a))*action_noise
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            # No slowmotion
            # time.sleep(timestep / speedup)
    # if animated and not always_return_paths:
    #     return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )

import matplotlib.pyplot as plt
from private_examples.point_mass_env import  PointMassEnv
from private_examples.point2D_env import Point2DEnv
from gym.envs.mujoco.reacher import ReacherEnv
from private_examples.reacher_env import get_fingertips
def plot_2D_path(traj, env):
    # Hacky
    try:
        inner_env = env._wrapped_env._wrapped_env
    except AttributeError:
        inner_env = env._wrapped_env.env.unwrapped
    if isinstance(inner_env, Point2DEnv):
        plt.xlim(env.observation_space.low[0], env.observation_space.high[0])
        plt.ylim(env.observation_space.low[1], env.observation_space.high[1])
    elif isinstance(inner_env, PointMassEnv):
        plt.xlim(*inner_env.boundary)
        plt.ylim(*inner_env.boundary)
    elif isinstance(inner_env, ReacherEnv):
        plt.xlim(-.3, .3)
        plt.ylim(-.3, .3)
        plt.scatter(*inner_env.goal, c='g', marker='x', s=100)
        traj = get_fingertips(traj[:, :2])
    else:
        return False
    plt.scatter(*inner_env.goal, c='g', marker='x', s=100)
    plt.plot(traj[:, 0], traj[:, 1], 'b', lw=2, zorder=1)
    plt.scatter(traj[:, 0], traj[:, 1], alpha=0.2, c='r', s=100, zorder=2)
    return True

import os.path
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+',
                        help='path to the snapshot files')
    parser.add_argument('--horizon', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--action_noise', type=float, default = 0.0)
    parser.add_argument('--no_query', action="store_true", default = False)
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    for f in args.files:
        with tf.Session() as sess:
            try:
                data = joblib.load(f)
            except:
                with tf.variable_scope("", reuse=True):
                    data = joblib.load(f)
            policy = data['policy']
            env = data['env']
            while True:
                path = rollout(env, policy, max_path_length=args.horizon,
                           animated=True, speedup=args.speedup, action_noise=args.action_noise)
                print(path['observations'])
                print(path['actions'])
                print(path['rewards'])
                print('cost: ', -np.sum(path['rewards']))
                is_plot = False
                is_plot = plot_2D_path(path['observations'], env)
                if args.no_query:
                    break
                else:
                    if not query_yes_no('Continue simulation?'):
                        break
            if is_plot:
                plt.savefig(os.path.join(os.path.dirname(f), 'trajectory-%.6f.png'%np.random.uniform()))