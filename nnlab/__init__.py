"""nnlab — small, readable NumPy implementations of classic neural-network algorithms.

The package backs the lesson notebooks in ``notebooks/``. Each module mirrors one
lesson and favours clarity over speed, so the code can be read side by side with
the equations.
"""

from . import activations, backprop, bam, fuzzy, learning_rules, mcculloch_pitts, style

__version__ = "1.0.0"

__all__ = ["activations", "backprop", "bam", "fuzzy", "learning_rules", "mcculloch_pitts", "style", "__version__"]
