"""Single-neuron learning rules: Hebbian, perceptron and delta (Widrow–Hoff).

All three rules work the same way. Present an input ``x``, compute
``net = w · x`` and an output ``o = f(net)``, then change the weights:

=================  ===========================================  ==============
Rule               Weight change ``Δw``                          Supervised?
=================  ===========================================  ==============
Hebbian            ``c · o · x``                                 no
Perceptron         ``c · (d − o) · x``                           yes
Delta              ``c · (d − o) · f'(net) · x``                 yes
=================  ===========================================  ==============

Each trainer returns a :class:`History` that records every single update so the
lessons can print the step-by-step tables that appear in textbooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from .activations import bipolar_sigmoid, bipolar_sigmoid_derivative, bipolar_step

__all__ = ["Step", "History", "hebbian", "perceptron", "delta_rule"]


@dataclass
class Step:
    """One weight update."""

    epoch: int
    sample: int
    net: float
    output: float
    target: Optional[float]
    delta_w: np.ndarray
    weights: np.ndarray  # weights *after* the update


@dataclass
class History:
    """Everything that happened during training."""

    initial_weights: np.ndarray
    steps: List[Step] = field(default_factory=list)
    epoch_errors: List[float] = field(default_factory=list)
    converged: bool = False

    @property
    def weights(self) -> np.ndarray:
        """Final weight vector."""
        return self.steps[-1].weights if self.steps else self.initial_weights

    @property
    def n_epochs(self) -> int:
        return self.steps[-1].epoch if self.steps else 0

    def weight_trajectory(self) -> np.ndarray:
        """Array of shape ``(n_steps + 1, n_weights)``, starting with the initial weights."""
        return np.vstack([self.initial_weights] + [s.weights for s in self.steps])

    def print_steps(self, precision: int = 3) -> None:
        """Pretty-print one line per update, textbook style."""
        fmt = lambda v: np.array2string(np.asarray(v), precision=precision, suppress_small=True)  # noqa: E731
        print(f"initial w = {fmt(self.initial_weights)}")
        for s in self.steps:
            target = "" if s.target is None else f"  d = {s.target:+.0f}"
            print(
                f"epoch {s.epoch}, x{s.sample + 1}:  net = {s.net:+.{precision}f}  "
                f"o = {s.output:+.{precision}f}{target}  ->  w = {fmt(s.weights)}"
            )


def _as_arrays(w0: Sequence[float], X: Sequence[Sequence[float]]):
    return np.asarray(w0, dtype=float).copy(), np.asarray(X, dtype=float)


def hebbian(
    w0: Sequence[float],
    X: Sequence[Sequence[float]],
    c: float = 1.0,
    activation: Callable = bipolar_step,
    epochs: int = 1,
) -> History:
    """Unsupervised Hebbian learning, ``Δw = c · f(w·x) · x``."""
    w, X = _as_arrays(w0, X)
    history = History(initial_weights=w.copy())
    for epoch in range(1, epochs + 1):
        for i, x in enumerate(X):
            net = float(w @ x)
            o = float(activation(net))
            dw = c * o * x
            w = w + dw
            history.steps.append(Step(epoch, i, net, o, None, dw, w.copy()))
    return history


def perceptron(
    w0: Sequence[float],
    X: Sequence[Sequence[float]],
    d: Sequence[float],
    c: float = 0.1,
    epochs: int = 1,
    stop_when_converged: bool = True,
) -> History:
    """Discrete perceptron rule, ``Δw = c · (d − sgn(w·x)) · x``.

    Training stops early once a full epoch makes no mistakes (if
    ``stop_when_converged``); ``History.converged`` reports whether that happened.
    """
    w, X = _as_arrays(w0, X)
    d = np.asarray(d, dtype=float)
    history = History(initial_weights=w.copy())
    for epoch in range(1, epochs + 1):
        mistakes = 0
        for i, (x, target) in enumerate(zip(X, d)):
            net = float(w @ x)
            o = float(bipolar_step(net))
            dw = c * (target - o) * x
            w = w + dw
            mistakes += int(o != target)
            history.steps.append(Step(epoch, i, net, o, float(target), dw, w.copy()))
        history.epoch_errors.append(float(mistakes))
        if mistakes == 0:
            history.converged = True
            if stop_when_converged:
                break
    return history


def delta_rule(
    w0: Sequence[float],
    X: Sequence[Sequence[float]],
    d: Sequence[float],
    c: float = 0.1,
    lam: float = 1.0,
    epochs: int = 1,
    tol: Optional[float] = None,
) -> History:
    r"""Continuous perceptron trained by gradient descent (the delta rule).

    Uses the bipolar sigmoid and the squared error
    :math:`E = \tfrac12 (d - o)^2`, giving
    :math:`\Delta w = c\,(d - o)\,\tfrac{\lambda}{2}(1 - o^2)\,x`.

    ``History.epoch_errors`` holds the total error accumulated during each epoch.
    If ``tol`` is given, training stops once that total drops below it.
    """
    w, X = _as_arrays(w0, X)
    d = np.asarray(d, dtype=float)
    history = History(initial_weights=w.copy())
    for epoch in range(1, epochs + 1):
        total_error = 0.0
        for i, (x, target) in enumerate(zip(X, d)):
            net = float(w @ x)
            o = float(bipolar_sigmoid(net, lam))
            total_error += 0.5 * (target - o) ** 2
            dw = c * (target - o) * float(bipolar_sigmoid_derivative(o, lam)) * x
            w = w + dw
            history.steps.append(Step(epoch, i, net, o, float(target), dw, w.copy()))
        history.epoch_errors.append(total_error)
        if tol is not None and total_error < tol:
            history.converged = True
            break
    return history
