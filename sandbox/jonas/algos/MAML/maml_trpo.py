

from sandbox.jonas.algos.MAML.maml_npo import MAMLNPO
from sandbox_maml.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class MAMLTRPO(MAMLNPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(MAMLTRPO, self).__init__(optimizer=optimizer, **kwargs)
