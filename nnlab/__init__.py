"""Small NumPy implementations of classic neural network algorithms.

This package is used by the lesson notebooks in ``notebooks/``. Each module matches
one lesson and is written to be read next to the equations, not to be fast.
"""

from . import activations, backprop, bam, fuzzy, learning_rules, mcculloch_pitts, style

__version__ = "1.0.0"

__all__ = ["activations", "backprop", "bam", "fuzzy", "learning_rules", "mcculloch_pitts", "style", "__version__"]
