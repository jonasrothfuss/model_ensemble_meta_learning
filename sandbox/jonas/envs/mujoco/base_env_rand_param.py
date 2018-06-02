from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab_maml.envs.mujoco.mujoco_env import MujocoEnv as MujocoEnvMAML
from sandbox.jonas.envs.helpers import get_all_function_arguments

from rllab.core.serializable import Serializable
import numpy as np
import warnings


class BaseEnvRandParams(Serializable):
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """

    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']


    def __init__(self, *args, log_scale_limit=2.0, fix_params=False, rand_params=RAND_PARAMS, random_seed=None, fixed_goal=True, max_path_length=0, **kwargs):
        """
        Half-Cheetah environment with randomized mujoco parameters
        :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
        :param random_seed: random seed for sampling the mujoco model params
        :param fix_params: boolean indicating whether the mujoco parameters shall be fixed
        :param rand_params: mujoco model parameters to sample
        """
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)

        self.log_scale_limit = log_scale_limit
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)
        self.fix_params = fix_params  # can be changed by calling the fix_mujoco_parameters method
        self.rand_params = rand_params
        self.fixed_goal = fixed_goal
        self.parameters_already_fixed = False
        self.n_steps = 0
        self.reward_range = None
        self.metadata = None
        if max_path_length is not None:
            self.max_path_length = max_path_length
        else:
            self.max_path_length = 10**8 #set to a large number

        args_all, kwargs_all = get_all_function_arguments(self.__init__, locals())
        Serializable.__init__(*args_all, **kwargs_all)

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

        # reset number of steps taken
        self.n_steps = 0

        # The first time reset is called -> sample and fix the mujoco parameters
        if self.fix_params and not self.parameters_already_fixed:
            self.sample_and_fix_parameters()

        if self.fix_params and reset_args is not None:
            warnings.warn("Environment parameters are fixed - reset_ars does not have any effect", UserWarning)

        # set mujoco model parameters
        elif (not self.fix_params) and (reset_args is not None):
            self.reset_mujoco_parameters(reset_args)
        elif not self.fix_params:
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
        for param, param_val in param_dict.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            setattr(self.model, param, param_val)

    def fix_parameters(self, param_dict):
        assert self.fix_params, "requires sample_and_fix_parameters to be True"
        self.parameters_already_fixed = True
        self.reset_mujoco_parameters(param_dict)

    def sample_and_fix_parameters(self):
        assert hasattr(self, 'sample_env_params'), "class must implement the sample_env_params method"
        assert self.fix_params, "requires sample_and_fix_parameters to be True"
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

            new_params = {}

            if 'body_mass' in self.rand_params:
                body_mass_multiplyers = np.array(1.5)**self.random_state.uniform(-log_scale_limit, log_scale_limit,  size=self.model.body_mass.shape)
                new_params['body_mass'] = self.model.body_mass * body_mass_multiplyers


            # body_inertia
            if 'body_inertia' in self.rand_params:
                body_inertia_multiplyers = np.array(1.5)**self.random_state.uniform(-log_scale_limit, log_scale_limit,  size=self.model.body_inertia.shape)
                new_params['body_inertia'] = body_inertia_multiplyers * self.model.body_inertia

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                dof_damping_multipliers = np.array(1.3)**self.random_state.uniform(-log_scale_limit, log_scale_limit, size=self.model.dof_damping.shape)
                new_params['dof_damping'] = np.multiply(self.model.dof_damping, dof_damping_multipliers)

            # friction at the body components
            if 'geom_friction' in self.rand_params:
                dof_damping_multipliers = np.array(1.5) ** self.random_state.uniform(-log_scale_limit, log_scale_limit,
                                                                                     size=self.model.geom_friction.shape)
                new_params['geom_friction'] = np.multiply(self.model.geom_friction, dof_damping_multipliers)

            param_sets.append(new_params)

        return param_sets