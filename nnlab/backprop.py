"""A two-layer (one hidden layer) network trained with error backpropagation.

The implementation is deliberately explicit — every line maps onto one
equation in Lesson 6 — rather than vectorised for speed.

Notation (Zurada 1992):

* ``V`` — input→hidden weights, shape ``(J, I)`` (the last input is a constant bias of 1)
* ``W`` — hidden→output weights, shape ``(J + 1,)`` (the hidden layer also gets a bias unit)
* ``y`` — hidden-layer outputs, ``o`` — network output
* ``δ_o = (d − o) f'(net_o)`` and ``δ_y = f'(net_y) · W δ_o``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .activations import bipolar_sigmoid, bipolar_sigmoid_derivative

__all__ = ["TwoLayerNet", "TrainingResult"]


@dataclass
class TrainingResult:
    epochs: int
    converged: bool
    epoch_errors: List[float] = field(default_factory=list)


class TwoLayerNet:
    """Single-output network with one hidden layer and bipolar sigmoid units.

    Parameters
    ----------
    V : array, shape (J, I)
        Initial input→hidden weights. Inputs are expected to already contain
        the bias component (e.g. ``[x1, x2, 1]``).
    W : array, shape (J + 1,)
        Initial hidden→output weights; the last entry is the hidden bias weight.
    lr : float
        Learning constant ``η``.
    lam : float
        Sigmoid steepness ``λ``.
    """

    def __init__(self, V: Sequence[Sequence[float]], W: Sequence[float], lr: float = 0.05, lam: float = 1.0):
        self.V = np.asarray(V, dtype=float).copy()
        self.W = np.asarray(W, dtype=float).copy()
        if self.W.shape != (self.V.shape[0] + 1,):
            raise ValueError("W must have one weight per hidden unit plus one bias weight")
        self.lr = lr
        self.lam = lam

    # ------------------------------------------------------------------ forward
    def hidden(self, x: Sequence[float]) -> np.ndarray:
        """Hidden-layer outputs with the bias unit (1) appended."""
        y = bipolar_sigmoid(self.V @ np.asarray(x, dtype=float), self.lam)
        return np.append(y, 1.0)

    def forward(self, x: Sequence[float]) -> float:
        return float(bipolar_sigmoid(self.W @ self.hidden(x), self.lam))

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Vectorised forward pass for a batch of inputs, shape ``(N, I)``."""
        X = np.asarray(X, dtype=float)
        Y = bipolar_sigmoid(X @ self.V.T, self.lam)
        Y = np.hstack([Y, np.ones((len(X), 1))])
        return bipolar_sigmoid(Y @ self.W, self.lam)

    # ----------------------------------------------------------------- training
    def train_step(self, x: Sequence[float], d: float) -> float:
        """Present one pattern, update both layers, return its error ``½(d − o)²``."""
        x = np.asarray(x, dtype=float)
        y = self.hidden(x)
        o = float(bipolar_sigmoid(self.W @ y, self.lam))

        delta_o = (d - o) * bipolar_sigmoid_derivative(o, self.lam)
        delta_y = self.W * delta_o * bipolar_sigmoid_derivative(y, self.lam)
        delta_y = delta_y[:-1]  # the bias unit has no incoming weights

        self.W = self.W + self.lr * delta_o * y
        self.V = self.V + self.lr * np.outer(delta_y, x)
        return 0.5 * (d - o) ** 2

    def fit(
        self,
        X: Sequence[Sequence[float]],
        d: Sequence[float],
        max_epochs: int = 10_000,
        tol: Optional[float] = 0.01,
    ) -> TrainingResult:
        """Online (pattern-by-pattern) training until the epoch error is below ``tol``."""
        result = TrainingResult(epochs=0, converged=False)
        for epoch in range(1, max_epochs + 1):
            E = sum(self.train_step(x, t) for x, t in zip(X, d))
            result.epoch_errors.append(float(E))
            result.epochs = epoch
            if tol is not None and E < tol:
                result.converged = True
                break
        return result
