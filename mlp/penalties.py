import numpy as np

seed = 22102017
rng = np.random.RandomState(seed)


class L1Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """

    def __init__(self, coefficient):
        """Create a new L1 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient * np.sum(np.abs(parameter))

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        # Gradient of |x| is sign(x), with 0 at x = 0
        return self.coefficient * np.sign(parameter)

    def __repr__(self):
        return 'L1Penalty({0})'.format(self.coefficient)


class L2Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        # L2 penalty: λ * 0.5 * ||w||^2
        return 0.5 * self.coefficient * np.sum(parameter ** 2)

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        # Gradient of 0.5 * λ * ||w||^2 is λ * w
        return self.coefficient * parameter

    def __repr__(self):
        return 'L2Penalty({0})'.format(self.coefficient)

class L1L2MixPenalty(object):
    """L1 & L2 mix penalty.
    """

    def __init__(self, coefficient, alpha):
        """Create a new L1 & L2 mix penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        assert 0 <= alpha <= 1., 'Alpha must be in [0, 1].'
        self.coefficient = coefficient
        self.alpha = alpha

    def __call__(self, parameter):
        """Calculate L1 & L2 mix penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        l1_term = np.sum(np.abs(parameter))
        l2_term = 0.5 * np.sum(parameter ** 2)
        return self.coefficient * (self.alpha * l1_term + (1 - self.alpha) * l2_term)

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        l1_grad = np.sign(parameter)
        l2_grad = parameter
        return self.coefficient * (self.alpha * l1_grad + (1 - self.alpha) * l2_grad)

    def __repr__(self):
        return 'L1L2MixPenalty(coefficient={0}, alpha={1})'.format(self.coefficient, self.alpha)
