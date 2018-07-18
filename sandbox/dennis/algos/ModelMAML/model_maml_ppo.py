from rllab_maml.misc import ext
from rllab_maml.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.ours.algos.ModelMAML.model_batch_maml_polopt import ModelBatchMAMLPolopt
from sandbox_maml.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer, MAMLPPOOptimizer
from sandbox_maml.rocky.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np

class ModelMAMLPPO(ModelBatchMAMLPolopt):
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
            init_kl_penalty=1,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(max_epochs=1)
            optimizer = MAMLPPOOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.use_maml = use_maml
        self.clip_eps = clip_eps
        self.target_inner_step = target_inner_step
        super(ModelMAMLPPO, self).__init__(**kwargs)
        self.kl_coeff = [init_kl_penalty] * self.meta_batch_size

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
        kl_list = []
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
                kl_list.append(tf.reduce_mean(dist.kl_sym(old_dist_info_vars[i], dist_info_vars)))
                new_params.append(params)
                lr = dist.likelihood_ratio_sym(action_vars[i], old_dist_info_vars[i], dist_info_vars)
                if self.entropy_bonus > 0:
                    entropy = self.entropy_bonus * tf.reduce_mean(dist.entropy_sym(dist_info_vars))
                else:
                    entropy = 0
                # formulate as a minimization problem
                # The gradient of the surrogate objective is the policy gradient
                surr_objs.append(- tf.reduce_mean(lr * adv_vars[i]) - entropy)

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
        kl_coeff_vars_list = list(tf.placeholder(tf.float32, shape=[], name='kl_%s' % i) for i in range(self.meta_batch_size))

        # MAML outer loop
        for i in range(self.meta_batch_size):
            dist_info_vars, _ = self.policy.updated_dist_info_sym(i, all_surr_objs[-1][i], obs_vars[i], params_dict=new_params[i])
            lr = dist.likelihood_ratio_sym(action_vars[i], old_dist_info_vars[i], dist_info_vars)
            if self.entropy_bonus > 0:
                entropy = self.entropy_bonus * tf.reduce_mean(dist.entropy_sym(dist_info_vars))
            else:
                entropy = 0
            kl_penalty = kl_list[i] * kl_coeff_vars_list[i]
            clipped_obj = tf.minimum(lr * adv_vars[i], tf.clip_by_value(lr, 1-self.clip_eps, 1+self.clip_eps) * adv_vars[i])
            surr_objs.append(- tf.reduce_mean(clipped_obj) - entropy + kl_penalty)

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
            inner_kl=kl_list,
            extra_inputs=kl_coeff_vars_list,
            meta_batch_size=self.meta_batch_size,
            num_grad_updates=self.num_grad_updates,
        )
        self.kl_list = kl_list
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
        kl_coeff = tuple(self.kl_coeff)
        if log: logger.log("Computing loss before")
        loss_before = self.optimizer.loss(input_list, extra_inputs=kl_coeff)
        if log: logger.log("Optimizing")    
        self.optimizer.optimize(input_list, extra_inputs=kl_coeff)
        if log: logger.log("Computing loss after")
        loss_after = self.optimizer.loss(input_list, extra_inputs=kl_coeff)

        if log: logger.log("Updating KL loss coefficients")
        sess = tf.get_default_session()
        kls = self.optimizer.inner_kl(input_list, extra_inputs=kl_coeff)
        for i, kl in enumerate(kls):
            if kl < self.target_inner_step / 1.5:
                self.kl_coeff[i] /= 2
            if kl > self.target_inner_step * 1.5:
                self.kl_coeff[i] *= 2

        if self.use_maml:
            if log: logger.record_tabular('LossBefore', loss_before)
            if log: logger.record_tabular('LossAfter', loss_after)
            if log: logger.record_tabular('dLoss', loss_before - loss_after)
            if log: logger.record_tabular('klDiff', np.mean(kls))
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            dynamics_model=self.dynamics_model,
            baseline=self.baseline,
            env=self.env,
        )
