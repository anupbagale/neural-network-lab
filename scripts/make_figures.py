"""Regenerate the README figure, docs/images/overview.png.

Usage:  python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402

from nnlab import activations as act  # noqa: E402
from nnlab.backprop import TwoLayerNet  # noqa: E402
from nnlab.bam import BAM, random_bipolar_pairs  # noqa: E402
from nnlab.learning_rules import delta_rule, hebbian, perceptron  # noqa: E402
from nnlab.mcculloch_pitts import GATES, truth_table, xor_network  # noqa: E402
from nnlab.style import PALETTE, apply_style  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
apply_style()

CORNERS = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])


def points(ax, outputs):
    for (x1, x2), out in zip(CORNERS, outputs):
        ax.scatter(x1, x2, s=120, zorder=3, marker="o" if out else "X",
                   color=PALETTE[0] if out else PALETTE[1], edgecolor="white", linewidth=1.2)


def finish(ax, title, xlabel="", ylabel=""):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# --------------------------------------------------------------- panels
def panel_gates(ax):
    g = np.linspace(-0.5, 1.5, 200)
    G1, G2 = np.meshgrid(g, g)
    for w, ls in [((1, -1), "--"), ((-1, 1), ":")]:
        ax.contour(G1, G2, w[0] * G1 + w[1] * G2, levels=[1], colors=[PALETTE[2]], linewidths=2, linestyles=ls)
    ax.contourf(G1, G2, (G1 - G2 >= 1) | (G2 - G1 >= 1), levels=[0.5, 1.5], colors=[PALETTE[2]], alpha=0.15)
    points(ax, [o for _, o in truth_table(xor_network)])
    ax.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), xticks=[0, 1], yticks=[0, 1], aspect="equal")
    finish(ax, "XOR with two threshold neurons", "$x_1$", "$x_2$")


def panel_activations(ax):
    net = np.linspace(-6, 6, 400)
    cols = colormaps["nnlab_seq"](np.linspace(0.3, 1, 4))
    for lam, c in zip([0.5, 1, 2, 6], cols):
        ax.plot(net, act.bipolar_sigmoid(net, lam), color=c, label=f"λ={lam}")
    ax.plot(net, act.bipolar_step(net), color=PALETTE[1], ls="--", lw=1.5, label="sgn")
    ax.legend(fontsize=8, loc="upper left")
    finish(ax, "Bipolar sigmoid", "net", "f(net)")


def panel_hebbian(ax):
    w0 = [1, -1, 0, 0.5]
    X = [[1, -2, 1.5, 0], [1, -0.5, -2, -1.5], [0, 1, -1, 1.5]]
    traj = hebbian(w0, X, c=1).weight_trajectory()
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i], marker="o", label=f"$w_{i + 1}$")
    ax.set_xticks(range(len(traj)))
    ax.legend(fontsize=8, ncol=2)
    finish(ax, "Hebbian learning", "update step", "weight")


def panel_perceptron(ax):
    rng = np.random.default_rng(0)
    n = 25
    pos = rng.normal((1.5, 1.5), 0.7, (n, 2))
    neg = rng.normal((-1.0, -0.5), 0.7, (n, 2))
    X = np.hstack([np.vstack([pos, neg]), np.ones((2 * n, 1))])
    d = np.r_[np.ones(n), -np.ones(n)]
    order = rng.permutation(2 * n)
    h = perceptron([1.0, -2.0, 1.5], X[order], d[order], c=0.1, epochs=50)
    ws = [h.initial_weights] + [s.weights for s in h.steps if s.sample == 2 * n - 1]
    shades = colormaps["nnlab_seq"](np.linspace(0.25, 1, len(ws)))
    xs = np.array([-3.5, 4])
    for w, c in zip(ws, shades):
        ax.plot(xs, -(w[0] * xs + w[2]) / w[1], color=c, lw=1.6)
    ax.scatter(*pos.T, s=22, color=PALETTE[0], edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(*neg.T, s=26, marker="X", color=PALETTE[1], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set(xlim=(-3.5, 4), ylim=(-3, 4))
    finish(ax, "Perceptron, boundary per epoch", "$x_1$", "$x_2$")


def panel_delta(ax):
    w0 = [1, -1, 0, 0.5]
    X = [[1, -2, 0, -1], [0, 1.5, -0.5, -1], [-1, 1, 0.5, -1]]
    d = [-1, -1, 1]
    for c in [0.05, 0.1, 0.5, 2.0]:
        ax.plot(delta_rule(w0, X, d, c=c, epochs=300).epoch_errors, label=f"c={c}")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    finish(ax, "Delta rule", "epoch", "error")


def xor_net():
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
    net = TwoLayerNet(V=[[1, -1, -1], [1, -2, 1]], W=[1, 1, -1], lr=0.05)
    net.fit(X, [0, 1, 1, 0])
    return net


def panel_backprop(ax, net):
    g = np.linspace(-0.5, 1.5, 250)
    G1, G2 = np.meshgrid(g, g)
    Z = net.predict(np.column_stack([G1.ravel(), G2.ravel(), np.ones(G1.size)])).reshape(G1.shape)
    ax.contourf(G1, G2, np.clip(Z, 0, 1), levels=np.linspace(0, 1, 11), cmap="nnlab_seq")
    for (x1, x2), t in zip(CORNERS, [0, 1, 1, 0]):
        ax.scatter(x1, x2, s=120, marker="o" if t else "X", color="white" if t else PALETTE[1],
                   edgecolor=PALETTE[1], linewidth=1.8, zorder=3)
    ax.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), xticks=[0, 1], yticks=[0, 1], aspect="equal")
    ax.grid(False)
    finish(ax, "Backpropagation on XOR", "$x_1$", "$x_2$")


def panel_bam(ax):
    rng = np.random.default_rng(42)
    ks = range(1, 26)
    frac = []
    for k in ks:
        ok = 0
        for _ in range(20):
            A, B = random_bipolar_pairs(k, 40, 40, rng)
            m = BAM.from_pairs(A, B)
            ok += np.mean([np.array_equal(m.forward(a), b) for a, b in zip(A, B)])
        frac.append(ok / 20)
    ax.plot(list(ks), frac, marker="o", markersize=4, color=PALETTE[0])
    ax.set_ylim(-0.03, 1.03)
    finish(ax, "BAM capacity (40 neurons)", "stored pairs", "recalled correctly")


def panel_fuzzy(ax):
    t = np.linspace(0, 40, 400)
    warm = np.clip(np.minimum((t - 15) / 7, (35 - t) / 7), 0, 1)
    ax.plot(t, warm, color=PALETTE[0], label="warm")
    ax.plot(t, 1 - warm, color=PALETTE[1], ls="--", label="not warm")
    ax.fill_between(t, np.minimum(warm, 1 - warm), color=PALETTE[2], alpha=0.35)
    ax.legend(fontsize=8, loc="center left")
    finish(ax, "Fuzzy sets", "temperature (°C)", "membership")


# --------------------------------------------------------------- output
def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", OUT / name)


if __name__ == "__main__":
    net = xor_net()
    panels = [panel_gates, panel_activations, panel_hebbian, panel_perceptron,
              panel_delta, lambda ax: panel_backprop(ax, net), panel_bam, panel_fuzzy]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, panel in zip(axes.flat, panels):
        panel(ax)
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    save(fig, "overview.png")

