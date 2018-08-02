import numpy as np
from collections import OrderedDict
from rllab_maml.misc import ext
from rllab_maml.core.serializable import Serializable
from sandbox.ignasi.policies.new_policy.base_mlp_policy import BaseMLPPolicy
from sandbox_maml.rocky.tf.misc import tensor_utils
import copy
import tensorflow as tf

load_params = True


class MAMLGaussianMLPPolicy(BaseMLPPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            learn_std=True,
            num_tasks=1,
            init_std=1.0,
            adaptive_std=False,
            bias_transform=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=tf.nn.tanh,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.identity,
            mean_network=None,
            std_network=None,
            std_parametrization='exp',
            grad_step_size=0.1,
            trainable_step_size=False,
            stop_grad=False,
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std: boolean indicating whether std shall be a trainable variable
        :param bias_transform: boolean indicating whether bias transformation shall be added to the MLP
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :param grad_step_size: (float) the step size taken in the learner's gradient update
        :param trainable_step_size: boolean indicating whether the inner grad_step_size shall be trainable
        :param stop_grad: whether or not to stop the gradient through the gradient.
        :param: parameter_space_noise: (boolean) whether parameter space noise shall be used when sampling from the policy
        """

        Serializable.quick_init(self, locals())

        BaseMLPPolicy.__init__(self,
                               name,
                               env_spec,
                               hidden_sizes=hidden_sizes,
                               learn_std=learn_std,
                               init_std=init_std,
                               adaptive_std=adaptive_std,
                               bias_transform=bias_transform,
                               std_share_network=std_share_network,
                               std_hidden_sizes=std_hidden_sizes,
                               min_std=min_std,
                               std_hidden_nonlinearity=std_hidden_nonlinearity,
                               hidden_nonlinearity=hidden_nonlinearity,
                               output_nonlinearity=output_nonlinearity,
                               mean_network=mean_network,
                               std_network=std_network,)

        self.stop_grad = stop_grad
        self.num_tasks = num_tasks
        self.all_fast_params_tensor = []
        self._all_param_gradients = []
        self.all_param_vals = None #[self.get_variable_values(self.all_params)] * num_tasks
        self.init_param_vals = None
        self.param_step_sizes = {}
        self.grad_step_size = grad_step_size
        self.trainable_step_size = trainable_step_size
        self._update_input_keys = ['observations', 'actions', 'advantages']

        with tf.variable_scope(self.name):

            # Create placeholders for the param weights of the different tasks
            self.all_params_ph = [OrderedDict([(key, tf.placeholder(tf.float32, shape=value.shape))
                                          for key, value in self.all_params.items()])
                             for _ in range(num_tasks)]

            # Create the variables for the inner learning rate
            for key, param in self.all_params.items():
                shape = param.get_shape().as_list()
                init_stepsize = np.ones(shape, dtype=np.float32) * self.grad_step_size
                self.param_step_sizes[key + "_step_size"] = tf.Variable(initial_value=init_stepsize,
                                                                        name='step_size_%s' % key,
                                                                        dtype=tf.float32,
                                                                        trainable=self.trainable_step_size)

            # compile the _cur_f_dist with updated params
            outputs = []
            with tf.variable_scope("post_updated_policy"):
                inputs = tf.split(self.input_tensor, self.num_tasks, 0)
                for i in range(self.num_tasks):
                    task_inp = inputs[i]
                    dist_info, _ = self.dist_info_sym(task_inp, dict(), all_params=self.all_params_ph[i], is_training=False)

                    outputs.append([dist_info['mean'], dist_info['log_std']])

                # TODO: Set a different name for this _cur_f_dist, so you can obtain actions w/o needing the params if
                # TODO: you aren't using the get_actions_batch (it'll be needed at test time when you are just evaluating
                # TODO: in one task)
                self._batch_cur_f_dist = tensor_utils.compile_function(
                    inputs=[self.input_tensor] +
                            sum([list(param_task_ph.values()) for param_task_ph in self.all_params_ph], []), # All the parameter values of the policy
                    outputs=outputs,
                )


    @property
    def update_input_keys(self):
        return self._update_input_keys

    def set_init_surr_obj(self, input_list, surr_objs_tensor):
        """ Set the surrogate objectives used the update the policy
        """
        self.input_list_for_grad = input_list
        self.surr_objs = surr_objs_tensor
        self._build_all_fast_params_tensor()

    def switch_to_init_dist(self):
        # switch cur policy distribution to pre-update policy
        init_param_vals = self.get_variable_values(self.all_params)
        self.all_param_vals = [init_param_vals] * self.num_tasks

    def get_actions_batch(self, observations):
        """
        :param observations: list of numpy arrays containing a batch of observations corresponding to a task -
                             shape of each numpy array must be (batch_size, ndim_obs)
        :return: actions - shape (batch_size * tasks, ndim_obs)
        """
        # assert that obs of all tasks have the same batch size
        batch_size = observations[0].shape[0]
        assert all([obs.shape[0] == batch_size for obs in observations])

        obs_stack = np.concatenate(observations, axis=0)
        result = self._batch_cur_f_dist(obs_stack,
                                        *sum([list(params_task.values()) for params_task in self.all_param_vals], []),
                                        )

        if len(result) == 2:
            # NOTE - this code assumes that there aren't 2 meta tasks in a batch
            means, log_stds = result
        else:
            means = np.concatenate([res[0] for res in result], axis=0)
            log_stds = np.concatenate([res[1] for res in result], axis=0)

        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def updated_dist_info_sym(self, task_id, surr_obj, new_obs_var, params_dict=None, is_training=True):
        """
        symbolically create MAML graph, for the meta-optimization, only called at the beginning of meta-training.
        Called more than once if you want to do more than one grad step.
        """
        old_params_dict = params_dict

        if old_params_dict is None:
            old_params_dict = self.all_params
        update_param_keys = params_dict.keys()

        grads = tf.gradients(surr_obj, [old_params_dict[key] for key in update_param_keys])
        if self.stop_grad:
            grads = [tf.stop_gradient(grad) for grad in grads]

        gradients = dict(zip(update_param_keys, grads))
        params_dict = dict(zip(update_param_keys, [
            old_params_dict[key] - tf.multiply(self.param_step_sizes[key + "_step_size"], gradients[key]) for key in
            update_param_keys]))

        return self.dist_info_sym(new_obs_var, all_params=params_dict, is_training=is_training)

    def compute_updated_dists(self, samples):
        """
        Compute fast gradients once per iteration and pull them out of tensorflow for sampling with the post-update policy.
        """
        sess = tf.get_default_session()
        num_tasks = len(samples)
        assert num_tasks == self.num_tasks
        input_list = list([] for _ in range(len(self.update_input_keys)))
        for i in range(num_tasks):
            inputs = ext.extract(samples[i], *self.update_input_keys)
            for j, input_name in enumerate(self.update_input_keys):
                if input_name == 'agent_infos':
                    input_list[j].extend([inputs[j][k] for k in self.distribution.dist_info_keys])
                else:
                    input_list[j].append(inputs[j])

        inputs = sum(input_list, [])

        feed_dict_inputs = list(zip(self.input_list_for_grad, inputs))
        feed_dict_params = list((self.all_params_ph[i][key], self.all_param_vals[i][key])
                                for i in range(num_tasks) for key in self.all_params_ph[0].keys())
        feed_dict = dict(feed_dict_inputs + feed_dict_params)
        self.all_param_vals, gradients = sess.run([self.all_fast_params_tensor, self._all_param_gradients], feed_dict=feed_dict)

    def _build_all_fast_params_tensor(self):
        update_param_keys = self.all_params.keys()
        with tf.variable_scope(self.name):
            # Create the symbolic graph for the one-step inner gradient update (It'll be called several times if
            # more gradient steps are needed
            for i in range(self.num_tasks):
                # compute gradients for a current task (symbolic)
                # for key in self.all_params.keys():
                #     tf.assign(self.all_params[key], self.all_params_ph[i][key])
                gradients = dict(zip(update_param_keys, tf.gradients(self.surr_objs[i],
                                                                     [self.all_params_ph[i][key] for key in update_param_keys]
                                                                     )))

                # gradient update for params of current task (symbolic)
                fast_params_tensor = OrderedDict(zip(update_param_keys,
                                                     [self.all_params_ph[i][key] - tf.multiply(
                                                         self.param_step_sizes[key + "_step_size"], gradients[key]) for
                                                      key in update_param_keys]))

                # tensors that represent the updated params for all of the tasks (symbolic)
                self.all_fast_params_tensor.append(fast_params_tensor)
                self._all_param_gradients.append(gradients)
