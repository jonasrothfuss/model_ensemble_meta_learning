import tensorflow as tf
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.ops import variables
GATE_OP = 1


class NoisyAdamOptimizer(object):

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, lamb=0.1, N=1e7, eta=1,
               use_locking=False, name="Noisy_Adam"):
    self._learning_rate = learning_rate
    self._sgd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, use_locking=use_locking)
    self._beta1 = beta1
    self._beta2 = beta2
    self._lamb = lamb
    self._eta  = eta
    self._N = N
    self._gamma = lamb/(N * eta)
    self.v = None
    self.m = None
    self.f = None
    self.mu = None
    self._init_variables = False
    self._k = tf.zeros(shape=(), name='step')

  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
    all_m, all_f = self.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    return self.apply_gradients(all_m, all_f, name=name)

  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    grads_and_vars_sgd = self._sgd_optimizer.compute_gradients(loss, var_list=var_list, gate_gradients=gate_gradients,
                                                          aggregation_method=aggregation_method,
                                                          colocate_gradients_with_ops=colocate_gradients_with_ops,
                                                          grad_loss=grad_loss)
    if not self._init_variables:
      self.mu = []
      self.f = []
      self.m = []
      for v in var_list:
        self.mu.append(tf.zeros_like(v))
        self.f.append(tf.zeros_like(v))
        self.m.append(tf.zeros_like(v))

    all_m, all_f = self._update_parameters(grads_and_vars_sgd)
    return all_m, all_f

  def _update_parameters(self, grads_and_vars_sgd):
    m = []
    f = []
    for i, (grad, var) in enumerate(grads_and_vars_sgd):
      v = grad - self._gamma * var
      m.append(self._beta1 * self.m[i] + (1 - self._beta2) * v)
      f.append(self._beta2 * self.f[i] + (1 - self._beta2) * tf.square(grad))
    return m, f

  def apply_gradients(self, all_m , all_f, name=None):
    cov = []
    for i, (m, f) in zip(all_m, all_f):
      tf.assign(self.m[i], m)
      tf.assign(self.f[i], f)
      tilde_m = m/(1 - self._beta1 ** self._k)
      hat_m = tilde_m/(tf.sqrt(f) + self._gamma)
      tf.assign(self.mu[i], self.mu[i] - self._learning_rate * hat_m)
      cov.append(self._lamb/(self._N * (f + self._gamma)))
    self._k += 1
    return self.mu, cov
