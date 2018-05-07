import numpy as np
from rllab.envs.base import Env
from rllab.core.serializable import Serializable


class ModelEnv(Env, Serializable):

    def __init__(self,
                 env,
                 dynamics_model):
        self._env = env
        self.dynamics_model = dynamics_model
        self.reward_fn = self._env.reward_np
        self.is_done = self._env.is_done
        self.current_obs = None
        super(ModelEnv, self).__init__()
        Serializable.__init__(self)

    def step(self, actions):
        next_obs = self.dynamics_model.predict(self.current_obs, actions)
        rewards = self.reward_fn(self.current_obs, actions, next_obs)
        dones = self.is_done(next_obs)
        self.current_obs = next_obs.copy()
        return next_obs, rewards, dones, {}

    def reset(self, init_pos=None):
        if init_pos is None:
            self.current_obs = np.zeros((1, self._env.observation_space.n))
        else:
            assert len(init_pos.shape) == 2
            self.current_obs = init_pos

    def action_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        return self._env.action_space

    @property
    def observation_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        return self._env.spec.observation_space

    @property
    def action_dim(self):
        return self._env.action_space.flat_dim

    def render(self):
        pass

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    def spec(self):
        return self._env.spec

    def horizon(self):
        return None

    def log_diagnostics(self, paths):
        pass
