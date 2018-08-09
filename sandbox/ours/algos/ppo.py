from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.ours.optimizers.ppo_optmizer import PPOOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class PPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            clip_eps=0.2,
            step_size=0.01,
            init_kl_penalty=1,
            entropy_coeff=0,
            adaptive_kl_penalty=True,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(max_epochs=10, batch_size=256, verbose=True)
            optimizer = PPOOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.clip_eps = clip_eps
        self.init_kl_penalty = init_kl_penalty
        self.adaptive_kl_penalty = adaptive_kl_penalty
        self.kl_coeff = init_kl_penalty
        self.entropy_coeff = entropy_coeff
        super(PPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        # entropy_bonus = sum(list(entropy_list[j][i] for j in range(self.num_grad_updates)))
        entropy = dist.entropy_sym(dist_info_vars)
        clipped_obj = tf.minimum(lr * advantage_var,
                                 tf.clip_by_value(lr, 1 - self.clip_eps, 1 + self.clip_eps) * advantage_var)

        if is_recurrent:
            mean_entropy = tf.reduce_sum(entropy) / tf.reduce_sum(valid_var)
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = - tf.reduce_sum(clipped_obj * valid_var) / tf.reduce_sum(valid_var) \
                        + self.kl_coeff * mean_kl - self.entropy_coeff * mean_entropy
        else:
            mean_entropy = tf.reduce_mean(entropy)
            mean_kl = tf.reduce_mean(kl)
            surr_loss = - tf.reduce_mean(clipped_obj) + self.kl_coeff * mean_kl - self.entropy_coeff * mean_entropy

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        extra_inputs = [tf.placeholder(tf.float32, shape=[], name='kl_coeff')]
        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            kl=mean_kl,
            inputs=input_list,
            extra_inputs=extra_inputs
            )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages",
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        kl_coeff = (self.kl_coeff,)

        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(all_input_values)
        logger.log("Optimizing")
        self.optimizer.optimize(all_input_values, extra_inputs=kl_coeff)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(all_input_values)

        logger.log("Computing KL")
        kl = self.optimizer.kl(all_input_values, extra_inputs=kl_coeff)
        if self.adaptive_kl_penalty:
            logger.log("Updating KL loss coefficients")
            if kl < self.step_size / 1.5:
                self.kl_coeff /= 2
            if kl > self.step_size * 1.5:
                self.kl_coeff *= 2

        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
