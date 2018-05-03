import numpy as np
from sandbox.jonas.controllers.base import Controller


class MPCcontroller(Controller):
    """ Model-Predictive Controller using random shooting method """

    def __init__(self,
                 env,
                 dynamics_model,
                 horizon=5,
                 reward_fn=None,
                 num_simulated_paths=1000,
                 ):
        """

        :param env: environment object (e.g. mujoco environment)
        :param dynamics_model: dynamics model
        :param horizon: number of timesteps to plan ahead
        :param reward_fn: cost function corresponding to the environment
        :param num_simulated_paths: number of paths simulated for the random shooting method
        """

        self.env = env
        self.dyn_model = dynamics_model
        self.horizon = horizon
        self.reward_fn = reward_fn
        self.num_simulated_paths = num_simulated_paths
        super().__init__()

    def get_action(self, state):
        

        actions_sequence = np.stack([np.stack([self.env.action_space.sample() for _ in range(self.num_simulated_paths)], axis=0)
                                for _ in range(self.horizon)], axis=0) # ndarray of shape (horizon, n_paths, ndim_action)

        states_t = np.tile(np.expand_dims(state, axis=0), (self.num_simulated_paths,1))
        states_sequence = np.empty((self.horizon + 1, self.num_simulated_paths, states_t.shape[-1]))
        states_sequence[0] = states_t

        assert actions_sequence.shape[0] == states_sequence.shape[0] - 1
        assert actions_sequence.shape[1] == states_sequence.shape[1]

        path_rewards = np.zeros(self.num_simulated_paths)
        for i in range(self.horizon):
            next_states = self.dyn_model.predict(states_sequence[i], actions_sequence[i])
            states_sequence[i+1] = next_states
            path_rewards += self.reward_fn(states_sequence[i], actions_sequence[i], states_sequence[i + 1])

        # select path with highest reward
        idx = np.argmax(path_rewards)
        best_action = actions_sequence[0, idx, :]
        assert best_action.shape == self.env.action_space.shape
        return best_action