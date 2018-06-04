import numpy as np
import pickle as pickle
from sandbox.rocky.tf.misc import tensor_utils
import copy


class MAMLModelVecEnvExecutor(object):
    def __init__(self, env, model, max_path_length, n_parallel):
        self.env = env
        self.model = model

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, 'wrapped_env'):
            self.unwrapped_env = self.unwrapped_env.wrapped_env

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        # check whether env has done function
        self.has_done_fn = hasattr(self.unwrapped_env, 'done')

        self.n_parallel = n_parallel
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self.ts = np.zeros(n_parallel, dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_n, traj_starting_obs=None):
        """
        :param action_n: batches of actions for all models/taks stacked on top of each other (n_models * batch_per_model, ndim_act)
        :return: predicted observations (n_models * batch_per_model, ndim_obs)
        """

        assert action_n.shape[0] == self.n_parallel

        # use the model to make (predicted) steps
        prev_obs = self.current_obs
        next_obs = self.model.predict_model_batches(prev_obs, action_n)
        rewards = self.unwrapped_env.reward(prev_obs, action_n, next_obs)

        if self.has_done_fn:
            dones = self.unwrapped_env.done(next_obs)
        else:
            dones = np.asarray([False] * self.n_parallel)

        env_infos = [{}] * self.n_parallel

        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        self.ts[dones] = 0

        if np.any(dones):
            if traj_starting_obs is None:
                next_obs[dones] = [self.env.reset() for _ in range(sum(dones))]
            else:
                idx = np.random.randint(low=0, high=traj_starting_obs.shape[0], size=sum(dones))
                next_obs[dones] = traj_starting_obs[idx, :]

        self.current_obs = next_obs

        return list(next_obs), list(rewards), list(dones), tensor_utils.stack_tensor_dict_list(env_infos) #lists

    def reset(self, traj_starting_obs=None):
        if traj_starting_obs is not None:
            results = [traj_starting_obs[np.random.randint(traj_starting_obs.shape[0]), :] for _ in range(self.n_parallel)] # randomly relect one observation of traj_starting_obs as initial obs
        else:
            results = [self.env.reset() for _ in range(self.n_parallel)] # get initial observation from environment
        self.current_obs = np.stack(results, axis=0)
        assert self.current_obs.ndim == 2
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return self.n_parallel

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass
