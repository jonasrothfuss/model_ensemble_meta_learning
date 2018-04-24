import numpy as np
import warnings

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides


class HalfCheetahEnvRandParams(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, log_scale_limit=2.0, fix_params=False, random_seed=None, **kwargs):
        """
        Hald Cheeta with randomized mujoco parameters
        :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
        :param random_seed: random seed for samling the mujoco model params
        """
        self.log_scale_limit = log_scale_limit
        self.random_state = np.random.RandomState(random_seed)
        self.fixed_params = False # can be changed by calling the fix_mujoco_parameters method

        super(HalfCheetahEnvRandParams, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def sample_env_params(self, num_param_sets, log_scale_limit=None):
        """
        generates randomized parameter sets for the mujoco env
        :param num_param_sets: number of parameter sets to obtain
        :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
        :return: array of length num_param_sets with dicts containing a randomized parameter set
        """

        if log_scale_limit is None:
            log_scale_limit = self.log_scale_limit

        param_sets = []
        for _ in range(num_param_sets):
            # body mass -> one multiplier for all body parts
            body_mass_multiplyer = np.array(2.0)**self.random_state.uniform(-log_scale_limit, log_scale_limit)
            new_body_mass = self.model.body_mass * body_mass_multiplyer

            # damping -> different multiplier for different dofs/joints
            dof_damping_multipliers = np.array(2.0)**self.random_state.uniform(-log_scale_limit, log_scale_limit,
                                                                       size=self.model.dof_damping.shape)
            new_dof_damping = np.multiply(self.model.dof_damping, dof_damping_multipliers)

            # body_inertia ->
            body_inertia_multiplyers = np.array(2.0)**self.random_state.uniform(-log_scale_limit, log_scale_limit,
                                                                        size=self.model.body_inertia.shape)
            new_body_inertia = np.multiply(self.model.body_inertia, body_inertia_multiplyers)

            param_sets.append({'body_mass': new_body_mass,
                               'dof_damping': new_dof_damping,
                               'body_inertia': new_body_inertia})

        return param_sets

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        """
        resets the environment and returns the initial observation
        :param init_state: initial state -> joint angles
        :param reset_args: dicts containing a randomized parameter set for altering the mujoco model params
                           if None, a new set of model params is sampled
        :return: initial observation
        """
        assert reset_args is None or type(reset_args) == dict, "reset_args must be a dict containing mujoco model params"


        if self.fixed_params and reset_args is not None:
            warnings.warn("Environment parameters are fixed - reset_ars does not have any effect", UserWarning)

        # set mujoco model parameters
        elif (not self.fixed_params) and (reset_args is not None):
            self.reset_mujoco_parameters(reset_args)
        elif not self.fixed_params:
            # sample parameter set
            reset_args = self.sample_env_params(1)[0]
            self.reset_mujoco_parameters(reset_args)

        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        run_cost = -1 * self.get_body_comvel("torso")[0]
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

    def reset_mujoco_parameters(self, param_dict):
        for param, param_val in param_dict.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            setattr(self.model, param, param_val)

    def fix_parameters(self, param_dict):
        self.reset_mujoco_parameters(param_dict)
        self.fixed_params = True

    def sample_and_fix_parameters(self):
        param_dict = self.sample_env_params(1)[0]
        self.fix_parameters(param_dict)
        return self
