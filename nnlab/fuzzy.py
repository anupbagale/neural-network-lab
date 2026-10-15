"""Fuzzy sets on a finite universe, with the standard (Zadeh) operations.

A fuzzy set assigns every element of the universe a *membership grade* in
``[0, 1]`` instead of a crisp in/out decision.  The standard operators are

* complement   ``μ_Ā(x) = 1 − μ_A(x)``
* union        ``μ_{A∪B}(x) = max(μ_A(x), μ_B(x))``
* intersection ``μ_{A∩B}(x) = min(μ_A(x), μ_B(x))``
* difference   ``μ_{A−B}(x) = min(μ_A(x), 1 − μ_B(x))``
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["FuzzySet", "check_properties"]


class FuzzySet:
    """A discrete fuzzy set.

    >>> A = FuzzySet([1, 2, 3], [0.0, 1.0, 0.5], name="A")
    >>> (A | ~A).membership
    array([1. , 1. , 0.5])
    """

    def __init__(self, universe: Sequence, membership: Sequence[float], name: str = "A"):
        self.universe = np.asarray(universe)
        self.membership = np.asarray(membership, dtype=float)
        if self.universe.shape != self.membership.shape:
            raise ValueError("universe and membership must have the same length")
        if np.any((self.membership < 0) | (self.membership > 1)):
            raise ValueError("membership grades must lie in [0, 1]")
        self.name = name

    def _check(self, other: "FuzzySet") -> None:
        if not np.array_equal(self.universe, other.universe):
            raise ValueError("fuzzy sets must share the same universe")

    def _new(self, membership: np.ndarray, name: str) -> "FuzzySet":
        return FuzzySet(self.universe, np.round(membership, 12), name)

    # Operators -----------------------------------------------------------
    def complement(self) -> "FuzzySet":
        return self._new(1.0 - self.membership, f"¬{self.name}")

    def union(self, other: "FuzzySet") -> "FuzzySet":
        self._check(other)
        return self._new(np.maximum(self.membership, other.membership), f"({self.name} ∪ {other.name})")

    def intersection(self, other: "FuzzySet") -> "FuzzySet":
        self._check(other)
        return self._new(np.minimum(self.membership, other.membership), f"({self.name} ∩ {other.name})")

    def difference(self, other: "FuzzySet") -> "FuzzySet":
        """``A − B = A ∩ ¬B``."""
        return self.intersection(other.complement())._renamed(f"({self.name} − {other.name})")

    def _renamed(self, name: str) -> "FuzzySet":
        self.name = name
        return self

    __invert__ = complement
    __or__ = union
    __and__ = intersection
    __sub__ = difference

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FuzzySet)
            and np.array_equal(self.universe, other.universe)
            and np.allclose(self.membership, other.membership)
        )

    def __repr__(self) -> str:
        pairs = ", ".join(f"{m:g}/{u}" for u, m in zip(self.universe, self.membership))
        return f"{self.name} = {{{pairs}}}"


def check_properties(A: FuzzySet, B: FuzzySet, C: FuzzySet) -> Dict[str, bool]:
    """Verify the classical set identities that *do* and *do not* survive fuzzification."""
    return {
        "commutative (∪)": (A | B) == (B | A),
        "commutative (∩)": (A & B) == (B & A),
        "associative (∪)": (A | (B | C)) == ((A | B) | C),
        "associative (∩)": (A & (B & C)) == ((A & B) & C),
        "distributive (∪ over ∩)": (A | (B & C)) == ((A | B) & (A | C)),
        "distributive (∩ over ∪)": (A & (B | C)) == ((A & B) | (A & C)),
        "De Morgan ¬(A ∪ B) = ¬A ∩ ¬B": ~(A | B) == (~A & ~B),
        "De Morgan ¬(A ∩ B) = ¬A ∪ ¬B": ~(A & B) == (~A | ~B),
        "involution ¬¬A = A": ~~A == A,
        "excluded middle A ∪ ¬A = X": np.allclose((A | ~A).membership, 1.0),
        "non-contradiction A ∩ ¬A = ∅": np.allclose((A & ~A).membership, 0.0),
    }
