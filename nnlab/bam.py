"""Bidirectional Associative Memory (Kosko, 1988).

A BAM stores pairs of bipolar patterns ``(a_k, b_k)`` in a single weight
matrix built with the Hebbian outer-product rule

.. math:: W = \\sum_k a_k^\\top b_k

and recalls one half of a pair from the other by bouncing activity back and
forth: ``b = sgn(a W)``, ``a = sgn(b Wᵀ)``, until neither side changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["BAM", "bipolar_threshold", "RecallTrace"]


def bipolar_threshold(net, previous: Optional[np.ndarray] = None) -> np.ndarray:
    """``+1`` where ``net > 0`` and ``-1`` where ``net < 0``.

    For ``net == 0`` the neuron keeps its previous state if one is given
    (Kosko's convention); otherwise it outputs ``-1``, matching the original lab.
    """
    net = np.asarray(net, dtype=float)
    out = np.where(net > 0, 1.0, -1.0)
    if previous is not None:
        out = np.where(net == 0, np.asarray(previous, dtype=float), out)
    return out


@dataclass
class RecallTrace:
    a: np.ndarray
    b: np.ndarray
    iterations: int
    energies: list


class BAM:
    """Bidirectional associative memory for bipolar (±1) patterns."""

    def __init__(self, n_a: int, n_b: int):
        self.W = np.zeros((n_a, n_b))

    @classmethod
    def from_pairs(cls, A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> "BAM":
        A, B = np.atleast_2d(A).astype(float), np.atleast_2d(B).astype(float)
        bam = cls(A.shape[1], B.shape[1])
        bam.store(A, B)
        return bam

    def store(self, A, B) -> None:
        """Add pattern pairs with the outer-product (Hebbian) rule."""
        for a, b in zip(np.atleast_2d(A), np.atleast_2d(B)):
            self.W += np.outer(a, b)

    def forward(self, a) -> np.ndarray:
        """One pass A → B."""
        return bipolar_threshold(np.asarray(a, dtype=float) @ self.W)

    def backward(self, b) -> np.ndarray:
        """One pass B → A."""
        return bipolar_threshold(np.asarray(b, dtype=float) @ self.W.T)

    def energy(self, a, b) -> float:
        """Lyapunov energy ``E(a, b) = −a W bᵀ``; it never increases during recall."""
        return float(-np.asarray(a) @ self.W @ np.asarray(b))

    def recall_from_b(self, b, max_iter: int = 50) -> RecallTrace:
        """Start recall from a (possibly noisy) B-layer cue."""
        return self.recall(self.backward(b), max_iter=max_iter)

    def recall(self, a, max_iter: int = 50) -> RecallTrace:
        """Iterate A → B → A … until a stable pair (a fixed point) is reached."""
        a = np.asarray(a, dtype=float)
        b = self.forward(a)
        energies = [self.energy(a, b)]
        for it in range(1, max_iter + 1):
            a_new = bipolar_threshold(b @ self.W.T, previous=a)
            b_new = bipolar_threshold(a_new @ self.W, previous=b)
            energies.append(self.energy(a_new, b_new))
            if np.array_equal(a_new, a) and np.array_equal(b_new, b):
                return RecallTrace(a, b, it, energies)
            a, b = a_new, b_new
        return RecallTrace(a, b, max_iter, energies)


def random_bipolar_pairs(k: int, n_a: int, n_b: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """``k`` random pattern pairs, used by the capacity experiment in Lesson 7."""
    return rng.choice([-1.0, 1.0], size=(k, n_a)), rng.choice([-1.0, 1.0], size=(k, n_b))


__all__.append("random_bipolar_pairs")
