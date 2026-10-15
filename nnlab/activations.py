"""Activation functions used throughout the lessons.

Every function accepts a scalar or a NumPy array and returns the same shape.
The steepness parameter ``lam`` (lambda) follows the notation of Zurada (1992):
larger values make the sigmoid steeper, and as ``lam -> inf`` the continuous
sigmoid approaches the discrete step function.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bipolar_sigmoid",
    "bipolar_sigmoid_derivative",
    "unipolar_sigmoid",
    "unipolar_sigmoid_derivative",
    "bipolar_step",
    "unipolar_step",
    "linear",
    "relu",
    "leaky_relu",
]


def bipolar_sigmoid(net, lam: float = 1.0):
    r"""Bipolar (hyperbolic-tangent-like) sigmoid, range :math:`(-1, 1)`.

    .. math:: f(net) = \frac{2}{1 + e^{-\lambda\,net}} - 1
    """
    net = np.asarray(net, dtype=float)
    return 2.0 / (1.0 + np.exp(-lam * net)) - 1.0


def bipolar_sigmoid_derivative(output, lam: float = 1.0):
    r"""Derivative of the bipolar sigmoid written in terms of its *output* ``o``.

    .. math:: f'(net) = \frac{\lambda}{2}\,(1 - o^2)
    """
    output = np.asarray(output, dtype=float)
    return 0.5 * lam * (1.0 - output**2)


def unipolar_sigmoid(net, lam: float = 1.0):
    r"""Unipolar (logistic) sigmoid, range :math:`(0, 1)`.

    .. math:: f(net) = \frac{1}{1 + e^{-\lambda\,net}}
    """
    net = np.asarray(net, dtype=float)
    return 1.0 / (1.0 + np.exp(-lam * net))


def unipolar_sigmoid_derivative(output, lam: float = 1.0):
    r"""Derivative of the unipolar sigmoid in terms of its output ``o``.

    .. math:: f'(net) = \lambda\,o\,(1 - o)
    """
    output = np.asarray(output, dtype=float)
    return lam * output * (1.0 - output)


def bipolar_step(net):
    """Bipolar signum: ``+1`` for ``net > 0``, ``-1`` for ``net < 0`` and ``0`` at ``net == 0``.

    This is NumPy's :func:`numpy.sign`; the zero case is kept on purpose because
    the textbook Hebbian and perceptron examples rely on it.
    """
    return np.sign(np.asarray(net, dtype=float))


def unipolar_step(net):
    """Unipolar step: ``1`` for ``net > 0``, ``0`` for ``net < 0`` and ``0.5`` at ``net == 0``."""
    return (bipolar_step(net) + 1.0) / 2.0


def linear(net):
    """Identity activation ``f(net) = net``."""
    return np.asarray(net, dtype=float)


def relu(net):
    """Rectified linear unit ``f(net) = max(0, net)``."""
    return np.maximum(0.0, np.asarray(net, dtype=float))


def leaky_relu(net, alpha: float = 0.1):
    """Leaky ReLU: ``net`` for positive inputs and ``alpha * net`` otherwise."""
    net = np.asarray(net, dtype=float)
    return np.where(net > 0, net, alpha * net)
