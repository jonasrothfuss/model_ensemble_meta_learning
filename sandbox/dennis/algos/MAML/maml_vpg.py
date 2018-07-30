# from rllab_maml.misc import ext
# from rllab_maml.misc.overrides import overrides
# import rllab.misc.logger as logger
# from sandbox.ours.algos.MAML.batch_maml_polopt import BatchMAMLPolopt
# from sandbox_maml.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer, MAMLPPOOptimizer
# from sandbox_maml.rocky.tf.misc import tensor_utils
# import tensorflow as tf
# import numpy as np

# class MAMLVPG(BatchMAMLPolopt):
#     """
#     Natural Policy Optimization.
#     """

#     def __init__(
#             self,
#             optimizer=None,
#             optimizer_args=None,
#             use_maml=True,
#             num_batches=10,
#             **kwargs):
#         if optimizer is None:
#             if optimizer_args is None:
#                 optimizer_args = dict(max_epochs=1, batch_size=num_batches, verbose=True)
#             optimizer = MAMLPPOOptimizer(**optimizer_args)
#         self.optimizer = optimizer
#         self.use_maml = use_maml
#         super(MAMLVPG, self).__init__(**kwargs)

#     def make_vars(self, stepnum='0'):
#         # lists over the meta_batch_size
#         obs_vars, action_vars, adv_vars = [], [], []
#         for i in range(self.meta_batch_size):
#             obs_vars.append(self.env.observation_space.new_tensor_variable(
#                 'obs' + stepnum + '_' + str(i),
#                 extra_dims=1,
#             ))
#             action_vars.append(self.env.action_space.new_tensor_variable(
#                 'action' + stepnum + '_' + str(i),
#                 extra_dims=1,
#             ))
#             adv_vars.append(tensor_utils.new_tensor(
#                 name='advantage' + stepnum + '_' + str(i),
#                 ndim=1, dtype=tf.float32,
#             ))
#         return obs_vars, action_vars, adv_vars

#     @overrides
#     def init_opt(self):
#         is_recurrent = int(self.policy.recurrent)
#         assert not is_recurrent  # not supported

#         dist = self.policy.distribution

#         old_dist_info_vars, old_dist_info_vars_list = [], []
#         for i in range(self.meta_batch_size):
#             old_dist_info_vars.append({
#                 k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s_%s' % (i, k))
#                 for k, shape in dist.dist_info_specs
#                 })
#             old_dist_info_vars_list += [old_dist_info_vars[i][k] for k in dist.dist_info_keys]

#         state_info_vars, state_info_vars_list = {}, []

#         all_surr_objs, input_list = [], []
#         new_params = None
#         for j in range(self.num_grad_updates):
#             obs_vars, action_vars, adv_vars = self.make_vars(str(j))
#             surr_objs = []

#             cur_params = new_params
#             new_params = []  # if there are several grad_updates the new_params are overwritten
#             kls = []

#             for i in range(self.meta_batch_size):
#                 if j == 0:
#                     dist_info_vars, params = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=self.policy.all_params)
#                 else:
#                     dist_info_vars, params = self.policy.updated_dist_info_sym(i, all_surr_objs[-1][i], obs_vars[i], params_dict=cur_params[i])

#                 new_params.append(params)
#                 logli = dist.log_likelihood_sym(action_vars[i], dist_info_vars)

#                 # formulate as a minimization problem
#                 # The gradient of the surrogate objective is the policy gradient
#                 surr_objs.append(- tf.reduce_mean(logli * adv_vars[i]))

#             input_list += obs_vars + action_vars + adv_vars + state_info_vars_list
#             if j == 0:
#                 # For computing the fast update for sampling
#                 self.policy.set_init_surr_obj(input_list, surr_objs)
#                 init_input_list = input_list

#             all_surr_objs.append(surr_objs)

#         obs_vars, action_vars, adv_vars = self.make_vars('test')
#         surr_objs = []
#         for i in range(self.meta_batch_size):
#             dist_info_vars, _ = self.policy.updated_dist_info_sym(i, all_surr_objs[-1][i], obs_vars[i], params_dict=new_params[i])

#             kl = dist.kl_sym(old_dist_info_vars[i], dist_info_vars)
#             kls.append(kl)
#             lr = dist.likelihood_ratio_sym(action_vars[i], old_dist_info_vars[i], dist_info_vars)
#             surr_objs.append(- tf.reduce_mean(lr*adv_vars[i]))

#         """ Sum over meta tasks """
#         surr_obj = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over meta_batch_size (the diff tasks)
#         input_list += obs_vars + action_vars + adv_vars + old_dist_info_vars_list

#         self.optimizer.update_opt(
#             loss=surr_obj,
#             target=self.policy,
#             inputs=input_list,
#             inner_kl=kls,
#             meta_batch_size=self.meta_batch_size,
#             num_grad_updates=self.num_grad_updates,
#         )
#         return dict()

#     @overrides
#     def optimize_policy(self, itr, all_samples_data, log=True):
#         assert len(all_samples_data) == self.num_grad_updates + 1  # we collected the rollouts to compute the grads and then the test!

#         if not self.use_maml:
#             all_samples_data = [all_samples_data[0]]

#         input_list = []
#         for step in range(len(all_samples_data)):  # these are the gradient steps
#             obs_list, action_list, adv_list = [], [], []
#             for i in range(self.meta_batch_size):

#                 inputs = ext.extract(
#                     all_samples_data[step][i],
#                     "observations", "actions", "advantages"
#                 )
#                 obs_list.append(inputs[0])
#                 action_list.append(inputs[1])
#                 adv_list.append(inputs[2])

#             input_list += obs_list + action_list + adv_list  # [ [obs_0], [act_0], [adv_0], [obs_1], ... ]

#         dist_info_list = []
#         for i in range(self.meta_batch_size):
#             agent_infos = all_samples_data[-1][i]['agent_infos']
#             dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
#         input_list += dist_info_list

#         if log: logger.log("Computing loss before")
#         loss_before = self.optimizer.loss(input_list)
#         if log: logger.log("Optimizing")    
#         self.optimizer.optimize(input_list)
#         if log: logger.log("Computing loss after")
#         loss_after = self.optimizer.loss(input_list)

#         kls = self.optimizer.inner_kl(input_list)

#         if self.use_maml:
#             if log: logger.record_tabular('LossBefore', loss_before)
#             if log: logger.record_tabular('LossAfter', loss_after)
#             if log: logger.record_tabular('dLoss', loss_before - loss_after)
#         return dict()

#     @overrides
#     def get_itr_snapshot(self, itr, samples_data):
#         return dict(
#             itr=itr,
#             policy=self.policy,
#             baseline=self.baseline,
#             env=self.env,
#         )

from rllab_maml.misc import ext
from rllab_maml.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.ours.algos.MAML.batch_maml_polopt import BatchMAMLPolopt
from sandbox_maml.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer, MAMLPPOOptimizer
from sandbox_maml.rocky.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np

class MAMLVPG(BatchMAMLPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            use_maml=True,
            clip_eps=0.2, 
            target_inner_step=0.01,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(max_epochs=1, batch_size=1, verbose=True)
            optimizer = MAMLPPOOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.use_maml = use_maml
        self.clip_eps = clip_eps
        self.target_inner_step = target_inner_step
        super(MAMLVPG, self).__init__(**kwargs)

    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        obs_vars, action_vars, adv_vars = [], [], []
        for i in range(self.meta_batch_size):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            action_vars.append(self.env.action_space.new_tensor_variable(
                'action' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            adv_vars.append(tensor_utils.new_tensor(
                name='advantage' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.float32,
            ))
        return obs_vars, action_vars, adv_vars

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        assert not is_recurrent  # not supported

        dist = self.policy.distribution   

        state_info_vars = {}

        all_surr_objs, input_list = [], []
        new_params = None
        # MAML inner loop
        for j in range(self.num_grad_updates):
            obs_vars, action_vars, adv_vars = self.make_vars(str(j))
            old_dist_info_vars, old_dist_info_vars_list = [], []
            for i in range(self.meta_batch_size):
                old_dist_info_vars.append({
                    k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s_%s_%s' % (j, i, k))
                    for k, shape in dist.dist_info_specs
                    })
                old_dist_info_vars_list += [old_dist_info_vars[i][k] for k in dist.dist_info_keys]
            surr_objs = []

            cur_params = new_params
            new_params = []  # if there are several grad_updates the new_params are overwritten

            for i in range(self.meta_batch_size):
                if j == 0:
                    dist_info_vars, params = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=self.policy.all_params)
                else:
                    dist_info_vars, params = self.policy.updated_dist_info_sym(i, all_surr_objs[-1][i], obs_vars[i], params_dict=cur_params[i])
                new_params.append(params)
                logli = dist.log_likelihood_sym(action_vars[i], dist_info_vars)
                lr = dist.likelihood_ratio_sym(action_vars[i], old_dist_info_vars[i], dist_info_vars)
                if self.entropy_bonus > 0:
                    entropy = self.entropy_bonus * tf.reduce_mean(dist.entropy_sym(dist_info_vars))
                else:
                    entropy = 0
                # formulate as a minimization problem
                # The gradient of the surrogate objective is the policy gradient
                surr_objs.append(- tf.reduce_mean(logli * adv_vars[i]) - entropy)

            input_list += obs_vars + action_vars + adv_vars + old_dist_info_vars_list
            if j == 0:
                # For computing the fast update for sampling
                self.policy.set_init_surr_obj(input_list, surr_objs)
                init_input_list = input_list

            all_surr_objs.append(surr_objs)

        obs_vars, action_vars, adv_vars = self.make_vars('test')
        old_dist_info_vars, old_dist_info_vars_list = [], []
        for i in range(self.meta_batch_size):
            old_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_test_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            old_dist_info_vars_list += [old_dist_info_vars[i][k] for k in dist.dist_info_keys]
        surr_objs = []

        # MAML outer loop
        for i in range(self.meta_batch_size):
            dist_info_vars, _ = self.policy.updated_dist_info_sym(i, all_surr_objs[-1][i], obs_vars[i], params_dict=new_params[i])
            logli = dist.log_likelihood_sym(action_vars[i], dist_info_vars)
            lr = dist.likelihood_ratio_sym(action_vars[i], old_dist_info_vars[i], dist_info_vars)
            if self.entropy_bonus > 0:
                entropy = self.entropy_bonus * tf.reduce_mean(dist.entropy_sym(dist_info_vars))
            else:
                entropy = 0
            surr_objs.append(- tf.reduce_mean(lr * adv_vars[i]) - entropy)

        if self.use_maml:
            surr_obj = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over meta_batch_size (the diff tasks)
            input_list += obs_vars + action_vars + adv_vars + old_dist_info_vars_list
        else:
            surr_obj = tf.reduce_mean(tf.stack(all_surr_objs[0], 0)) # if not meta, just use the first surr_obj
            input_list = init_input_list
        self.optimizer.update_opt(
            loss=surr_obj,
            target=self.policy,
            inputs=input_list,
            inner_kl=None,
            meta_batch_size=self.meta_batch_size,
            num_grad_updates=self.num_grad_updates,
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, all_samples_data, log=True):
        assert len(all_samples_data) == self.num_grad_updates + 1  # we collected the rollouts to compute the grads and then the test!

        if not self.use_maml:
            all_samples_data = [all_samples_data[0]]

        input_list = []
        for step in range(len(all_samples_data)):  # these are the gradient steps
            obs_list, action_list, adv_list, dist_info_list = [], [], [], []
            for i in range(self.meta_batch_size):

                inputs = ext.extract(
                    all_samples_data[step][i],
                    "observations", "actions", "advantages"
                )
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                adv_list.append(inputs[2])
                agent_infos = all_samples_data[step][i]['agent_infos']
                dist_info_list.extend([agent_infos[k] for k in self.policy.distribution.dist_info_keys])

            input_list += obs_list + action_list + adv_list + dist_info_list  # [ [obs_0], [act_0], [adv_0], [dist_0], [obs_1], ... ]
        if log: logger.log("Computing loss before")
        loss_before = self.optimizer.loss(input_list)
        if log: logger.log("Optimizing")    
        self.optimizer.optimize(input_list)
        if log: logger.log("Computing loss after")
        loss_after = self.optimizer.loss(input_list)


        if self.use_maml:
            if log: logger.record_tabular('LossBefore', loss_before)
            if log: logger.record_tabular('LossAfter', loss_after)
            if log: logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

