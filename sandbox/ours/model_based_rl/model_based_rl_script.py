from rllab.envs.own_envs.point_2d_env import PointEnv
from sandbox.ours.dynamics import MLPDynamicsModel

from sandbox.ours.controllers import RandomController, MPCcontroller
from sandbox.ours.model_based_rl.helpers import sample, path_reward
from pprint import pprint

import tensorflow as tf
import os
import numpy as np



def reward_fn_point_env(state, action, new_state):
    if new_state.ndim == 2:
        assert new_state.shape[1] == 2
        return - np.sum(new_state**2, axis=1)
    elif new_state.ndim == 1:
        assert new_state.shape[0] == 2
        return - np.sum(new_state ** 2)
    else:
        raise AssertionError("state must be numpy array with ndim = 1 or ndim == 2")



def train(env,
          reward_fn,
          render=False,
          learning_rate=1e-3,
          onpol_iters=10,
          batch_size=200,
          dynamics_iters=60,
          num_paths_random=1000,
          num_paths_onpol=10,
          num_simulated_paths=10000,
          env_horizon=1000,
          mpc_horizon=15,

          ):
    """

    Arguments:

    onpol_iters                 Number of iterations of onpolicy aggregation for the loop to run.

    dynamics_iters              Number of iterations of training for the dynamics model
    |_                          which happen per iteration of the aggregation loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated
    |                           by a random agent. We use these to train our
    |_                          initial dynamics model.

    num_paths_onpol             Number of paths to collect at each iteration of
    |_                          aggregation, using the Model Predictive Control policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining
    |                           fictitious rollouts, and picking the first action
    |                           of the best fictitious rollout. This argument is
    |                           how many timesteps should be in each fictitious
    |_                          rollout.

    n_layers/size/activations   Neural network architecture arguments.

    """


    # collect initial data with a random controller

    env.reset()

    random_controller = RandomController(env)

    random_paths = sample(env, random_controller, num_paths=num_paths_random, horizon=env_horizon)
    print("Collected {} paths with random policy".format(len(random_paths)))


    # Build dynamics model and MPC controllers

    sess = tf.Session()

    dynamics_model = MLPDynamicsModel("dynamics_model", env, hidden_sizes=(32, 32), hidden_nonlinearity=tf.nn.relu, batch_size=batch_size)

    mpc_controller = MPCcontroller(env=env,
                                   dynamics_model=dynamics_model,
                                   horizon=mpc_horizon,
                                   reward_fn=reward_fn,
                                   num_simulated_paths=num_simulated_paths)

    # Tensorflow session building

    sess.__enter__()
    tf.global_variables_initializer().run()


    # Multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset
    # and then taking on-policy samples and aggregating to the dataset.

    dataset = random_paths
    for itr in range(onpol_iters):
        # fit dynamics model

        obs = np.concatenate([path['observations'] for path in dataset], axis=0)
        obs_next = np.concatenate([path['next_observations'] for path in dataset], axis=0)
        act = np.concatenate([path['actions'] for path in dataset], axis=0)

        dynamics_model.fit(obs, act, obs_next, verbose=True, epochs=10)

        # generate on-policy data
        new_data_rl = sample(env, mpc_controller, num_paths=num_paths_onpol, horizon=env_horizon, verbose=True)

        # aggregate data
        dataset.extend(new_data_rl)

        # calculate cost and returns
        returns = np.concatenate([path['rewards'] for path in new_data_rl], axis=0)
        rewards = [path_reward(reward_fn, path) for path in new_data_rl]

        # LOGGING
        # Statistics for performance of MPC policy using
        # our learned dynamics model
        print('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        print('AverageRewards', np.mean(rewards))
        print('StdRewards', np.std(rewards))
        print('MinimumRewards', np.min(rewards))
        print('MaximumRewards', np.max(rewards))
        # In terms of true environment reward of your rolled out trajectory using the MPC controller
        print('AverageReturn', np.mean(returns))
        print('StdReturn', np.std(returns))
        print('MinimumReturn', np.min(returns))
        print('MaximumReturn', np.max(returns))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='PointEnv')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=15)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=60)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=1000) #TODO change back to 10000
    parser.add_argument('--onpol_paths', '-d', type=int, default=10)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=10)  #TODO change back to 1000
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make env
    if args.env_name is "PointEnv":
        env = PointEnv()
        reward_fn = reward_fn_point_env

    train(env=env,
          reward_fn=reward_fn,
          render=args.render,
          learning_rate=args.learning_rate,
          onpol_iters=args.onpol_iters,
          dynamics_iters=args.dyn_iters,
          batch_size=args.batch_size,
          num_paths_random=args.random_paths,
          num_paths_onpol=args.onpol_paths,
          num_simulated_paths=args.simulated_paths,
          env_horizon=args.ep_len,
          mpc_horizon=args.mpc_horizon,
          )


if __name__ == "__main__":
    main()




