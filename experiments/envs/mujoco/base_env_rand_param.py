from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv

import numpy as np
import warnings


class BaseEnvRandParams:
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """

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

    def reset_mujoco_parameters(self, param_dict):
        assert isinstance(self, MujocoEnv), "Must be a Mujoco Environment"
        for param, param_val in param_dict.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            setattr(self.model, param, param_val)

    def fix_parameters(self, param_dict):
        self.reset_mujoco_parameters(param_dict)
        self.fixed_params = True

    def sample_and_fix_parameters(self):
        assert hasattr(self, 'sample_env_params'), "class must implement the sample_env_params method"
        param_dict = self.sample_env_params(1)[0]
        self.fix_parameters(param_dict)
        return self

    def sample_env_params(self, num_param_sets, log_scale_limit=None):
        """
        generates randomized parameter sets for the mujoco env
        :param num_param_sets: number of parameter sets to obtain
        :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
        :return: array of length num_param_sets with dicts containing a randomized parameter set
        """
        assert hasattr(self, 'random_state'), "random_state must be set in the constructor"

        if log_scale_limit is None:
            log_scale_limit = self.log_scale_limit

        param_sets = []
        for _ in range(num_param_sets):
            # body mass -> one multiplier for all body parts
            body_mass_multiplyer = np.array(2.0)**self.random_state.uniform(-log_scale_limit, log_scale_limit)
            new_body_mass = self.model.body_mass * body_mass_multiplyer

            # body_inertia ->
            body_inertia_multiplyer = np.array(2.0)**self.random_state.uniform(-log_scale_limit, log_scale_limit)
            new_body_inertia = body_inertia_multiplyer * self.model.body_inertia

            # damping -> different multiplier for different dofs/joints
            dof_damping_multipliers = np.array(2.0)**self.random_state.uniform(-log_scale_limit, log_scale_limit,
                                                                       size=self.model.dof_damping.shape)
            new_dof_damping = np.multiply(self.model.dof_damping, dof_damping_multipliers)

            param_sets.append({'body_mass': new_body_mass,
                               'body_inertia': new_body_inertia,
                               'dof_damping': new_dof_damping,})

        return param_sets