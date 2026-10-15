"""Plot settings shared by all figures in the lab."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

#: Categorical colours, always used in this order (validated for colour-vision deficiency).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

#: Single-hue sequential ramp (light to dark blue) for magnitudes, e.g. λ sweeps.
SEQUENTIAL = LinearSegmentedColormap.from_list("nnlab_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
#: Diverging blue ↔ red with a neutral grey midpoint, e.g. signed weights.
DIVERGING = LinearSegmentedColormap.from_list("nnlab_div", ["#2a78d6", "#f0efec", "#e34948"])


def apply_style() -> None:
    """Set matplotlib rcParams for clean, recessive axes and the lab palette."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.frameon": False,
            "text.color": INK,
            "image.cmap": "nnlab_seq",
        }
    )
    for cmap in (SEQUENTIAL, DIVERGING):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap)
