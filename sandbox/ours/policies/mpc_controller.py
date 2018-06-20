from sandbox.rocky.tf.policies.base import Policy
from rllab.core.serializable import Serializable
import numpy as np

class MPCController(Policy, Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            discount=1,
            n_candidates=1000,
            horizon=10,
    ):
        self.counter = 0
        self.obs = []
        self.pred_obs = []
        self.actions = []
        self.dynamics_model = dynamics_model
        self.discount = discount
        self.env = env
        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, 'wrapped_env'):
            self.unwrapped_env = self.unwrapped_env.wrapped_env

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"
        self._n_candidates = n_candidates
        self.horizon = horizon
        Serializable.quick_init(self, locals())
        super(MPCController, self).__init__(env_spec=env.spec)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]
        action = self.get_best_action(observation)
        return action, dict()

    def get_actions(self, observations):
        actions = self.get_best_action(observations)
        return actions, dict()

    def get_random_action(self, n):
        return self.action_space.sample_n(n)

    def get_best_action(self, observation):
        n = self._n_candidates
        m = len(observation)
        h = self.horizon
        returns = np.zeros((n * m, ))

        a = self.get_random_action(h * n * m).reshape((h, n * m, -1))

        for t in range(h):
            if t == 0:
                cand_a = a[t].reshape((m, n, -1))
                observation = np.repeat(observation, n, axis=0)
            next_observation = self.dynamics_model.predict(observation, a[t])
            rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation
        returns = returns.reshape(m, n)
        return cand_a[range(m), np.argmax(returns, axis=1)]

    #################################################################

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        pass

