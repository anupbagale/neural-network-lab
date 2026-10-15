"""McCulloch–Pitts threshold neurons and the logic gates built from them."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["MPNeuron", "GATES", "XOR_HIDDEN", "XOR_OUTPUT", "xor_network", "truth_table"]


@dataclass(frozen=True)
class MPNeuron:
    """A McCulloch–Pitts neuron.

    The neuron computes the weighted sum ``net = w · x`` and fires (outputs 1)
    when ``net >= threshold``. Setting ``inverted=True`` flips the output, which
    is how the original lab builds NOT, NAND and NOR from AND/OR thresholds.
    """

    weights: Tuple[float, ...]
    threshold: float
    inverted: bool = False

    def net(self, x: Sequence[float]) -> float:
        return float(np.dot(self.weights, x))

    def __call__(self, *x: float) -> int:
        fires = self.net(x) >= self.threshold
        return int(fires != self.inverted)


#: Single-neuron gates exactly as configured in the original lab notebook.
GATES: Dict[str, MPNeuron] = {
    "AND": MPNeuron(weights=(1, 1), threshold=2),
    "OR": MPNeuron(weights=(1, 1), threshold=1),
    "NOT": MPNeuron(weights=(1,), threshold=1, inverted=True),
    "NAND": MPNeuron(weights=(1, 1), threshold=2, inverted=True),
    "NOR": MPNeuron(weights=(1, 1), threshold=1, inverted=True),
}

# XOR needs two layers: z1 = x1 AND NOT x2, z2 = x2 AND NOT x1, y = z1 OR z2.
XOR_HIDDEN = (MPNeuron(weights=(1, -1), threshold=1), MPNeuron(weights=(-1, 1), threshold=1))
XOR_OUTPUT = MPNeuron(weights=(1, 1), threshold=1)


def xor_network(x1: int, x2: int) -> int:
    """Two-layer McCulloch–Pitts network that computes XOR."""
    z = [neuron(x1, x2) for neuron in XOR_HIDDEN]
    return XOR_OUTPUT(*z)


def truth_table(gate: Callable[..., int], n_inputs: int = 2) -> List[Tuple[Tuple[int, ...], int]]:
    """Return ``[(inputs, output), ...]`` for every binary input combination."""
    return [(inputs, gate(*inputs)) for inputs in product((0, 1), repeat=n_inputs)]
