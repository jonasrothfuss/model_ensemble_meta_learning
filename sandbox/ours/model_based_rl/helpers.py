import numpy as np

def sample(env,
           controller,
           num_paths=10,
           horizon=1000,
           render=False,
           verbose=False):
    """
        Samples paths in a environment with a provided controller
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    paths = []

    for i_episode in range(num_paths):
        state = env.reset()
        state_t_array, action_array, state_t1_array, reward_array = [], [], [], []

        for t in range(horizon):
            if render and (i_episode % 10 == 0):
                env.render()
            action = controller.get_action(state)
            prev_state = state
            state, reward, done, info = env.step(action)

            # append triples (obs_t, act_t, obs_t1) to dataset
            state_t_array.append(prev_state)
            action_array.append(action)
            state_t1_array.append(state)
            reward_array.append(reward)

            if done:
                break

        if verbose:
            print("Generated new episode {} - Length: {}, Mean-Reward: {}".format(i_episode, len(reward_array),
                                                                                  np.mean(reward_array)))

        path_dict = {
            'observations': np.stack(state_t_array, axis=0),
            'next_observations': np.stack(state_t1_array, axis=0),
            'actions': np.stack(action_array, axis=0),
            'rewards': np.stack(reward_array, axis=0)

        }
        paths.append(path_dict)

    return paths


def path_reward(reward_fn, path):
    return trajectory_cost_fn(reward_fn, path['observations'], path['actions'], path['next_observations'])

def trajectory_cost_fn(reward_fn, states, actions, next_states):
    trajectory_cost = 0
    for i in range(len(actions)):
        trajectory_cost += reward_fn(states[i], actions[i], next_states[i])
    return trajectory_cost