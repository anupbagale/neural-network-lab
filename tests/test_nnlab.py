"""Regression tests: every algorithm must reproduce the hand-worked lab results."""

import numpy as np
import pytest

from nnlab import activations as act
from nnlab.backprop import TwoLayerNet
from nnlab.bam import BAM, bipolar_threshold
from nnlab.fuzzy import FuzzySet, check_properties
from nnlab.learning_rules import delta_rule, hebbian, perceptron
from nnlab.mcculloch_pitts import GATES, truth_table, xor_network

# Training data used in Lessons 3–5 (Zurada 1992, Chapter 2 worked examples)
W0 = [1, -1, 0, 0.5]
X_HEBB = [[1, -2, 1.5, 0], [1, -0.5, -2, -1.5], [0, 1, -1, 1.5]]
X_SUP = [[1, -2, 0, -1], [0, 1.5, -0.5, -1], [-1, 1, 0.5, -1]]
D_SUP = [-1, -1, 1]


# ---------------------------------------------------------------- activations
def test_sigmoids_at_zero_and_limits():
    assert act.bipolar_sigmoid(0) == pytest.approx(0)
    assert act.unipolar_sigmoid(0) == pytest.approx(0.5)
    assert act.bipolar_sigmoid(50) == pytest.approx(1)
    assert act.unipolar_sigmoid(-50) == pytest.approx(0, abs=1e-12)


def test_lambda_changes_steepness():
    """Regression for the original plotting bug where λ was ignored."""
    assert act.bipolar_sigmoid(1, lam=5) > act.bipolar_sigmoid(1, lam=1)


@pytest.mark.parametrize("lam", [0.5, 1.0, 3.0])
def test_sigmoid_derivatives_match_finite_differences(lam):
    net, h = np.linspace(-4, 4, 41), 1e-6
    for f, df in [
        (act.bipolar_sigmoid, act.bipolar_sigmoid_derivative),
        (act.unipolar_sigmoid, act.unipolar_sigmoid_derivative),
    ]:
        numeric = (f(net + h, lam) - f(net - h, lam)) / (2 * h)
        np.testing.assert_allclose(df(f(net, lam), lam), numeric, atol=1e-6)


def test_piecewise_activations():
    x = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_array_equal(act.bipolar_step(x), [-1, 0, 1])
    np.testing.assert_array_equal(act.unipolar_step(x), [0, 0.5, 1])
    np.testing.assert_array_equal(act.relu(x), [0, 0, 3])
    np.testing.assert_allclose(act.leaky_relu(x), [-0.2, 0, 3])
    np.testing.assert_array_equal(act.linear(x), x)


# ------------------------------------------------------------ McCulloch–Pitts
@pytest.mark.parametrize(
    "name, expected",
    [
        ("AND", [0, 0, 0, 1]),
        ("OR", [0, 1, 1, 1]),
        ("NAND", [1, 1, 1, 0]),
        ("NOR", [1, 0, 0, 0]),
    ],
)
def test_two_input_gates(name, expected):
    assert [out for _, out in truth_table(GATES[name])] == expected


def test_not_gate():
    assert [out for _, out in truth_table(GATES["NOT"], n_inputs=1)] == [1, 0]


def test_xor_network():
    assert [out for _, out in truth_table(xor_network)] == [0, 1, 1, 0]


# ------------------------------------------------------------- learning rules
def test_hebbian_matches_hand_calculation():
    h = hebbian(W0, X_HEBB, c=1)
    assert [s.net for s in h.steps] == pytest.approx([3.0, -0.25, -3.0])
    np.testing.assert_allclose(h.weights, [1, -3.5, 4.5, 0.5])


def test_perceptron_matches_hand_calculation():
    h = perceptron(W0, X_SUP, D_SUP, c=0.1, epochs=1)
    np.testing.assert_allclose(h.steps[0].weights, [0.8, -0.6, 0, 0.7])
    np.testing.assert_allclose(h.weights, [0.6, -0.4, 0.1, 0.5])


def test_perceptron_converges_on_separable_data():
    h = perceptron(W0, X_SUP, D_SUP, c=0.1, epochs=50)
    assert h.converged
    outputs = np.sign(np.asarray(X_SUP) @ h.weights)
    np.testing.assert_array_equal(outputs, D_SUP)


def test_delta_rule_one_epoch():
    h = delta_rule(W0, X_SUP, D_SUP, c=0.1, epochs=1)
    np.testing.assert_allclose(h.weights, [0.94737155, -0.92963027, 0.01608141, 0.50464886], atol=1e-7)


def test_delta_rule_error_decreases():
    h = delta_rule(W0, X_SUP, D_SUP, c=0.1, epochs=2000, tol=0.01)
    assert h.converged
    assert h.epoch_errors[-1] < h.epoch_errors[0]


# ---------------------------------------------------------------- backprop
XOR_X = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
XOR_D = [0, 1, 1, 0]


def make_xor_net():
    return TwoLayerNet(V=[[1, -1, -1], [1, -2, 1]], W=[1, 1, -1], lr=0.05)


def test_backprop_reproduces_original_lab():
    net = make_xor_net()
    result = net.fit(XOR_X, XOR_D, max_epochs=10_000, tol=0.01)
    assert result.converged and result.epochs == 4494
    np.testing.assert_allclose(net.W, [2.951527, -2.97163033, 3.96064113], atol=1e-6)
    np.testing.assert_allclose(net.V, [[3.0740777, -3.200431, -1.53326459], [3.06012978, -3.17927401, 1.67173102]], atol=1e-6)


def test_backprop_solves_xor():
    net = make_xor_net()
    net.fit(XOR_X, XOR_D)
    np.testing.assert_array_equal(np.round(net.predict(XOR_X)), XOR_D)


def test_backprop_gradient_matches_numerical_gradient():
    """δ-rule updates must equal −η ∂E/∂w computed by finite differences."""
    x, d, h = np.array([1.0, 0.0, 1.0]), 1.0, 1e-6

    def error(V, W):
        return 0.5 * (d - TwoLayerNet(V, W).forward(x)) ** 2

    base = make_xor_net()
    V0, W0_ = base.V.copy(), base.W.copy()
    base.train_step(x, d)
    dW, dV = (base.W - W0_) / base.lr, (base.V - V0) / base.lr

    for i in range(W0_.size):
        Wp, Wm = W0_.copy(), W0_.copy()
        Wp[i] += h
        Wm[i] -= h
        assert dW[i] == pytest.approx(-(error(V0, Wp) - error(V0, Wm)) / (2 * h), abs=1e-6)
    for idx in np.ndindex(V0.shape):
        Vp, Vm = V0.copy(), V0.copy()
        Vp[idx] += h
        Vm[idx] -= h
        assert dV[idx] == pytest.approx(-(error(Vp, W0_) - error(Vm, W0_)) / (2 * h), abs=1e-6)


# ------------------------------------------------------------------- fuzzy
X_U = [1, 2, 3, 4, 5]
A = FuzzySet(X_U, [0, 1, 0.5, 0.3, 0.2], "A")
B = FuzzySet(X_U, [0, 0.5, 0.7, 0.2, 0.4], "B")
C = FuzzySet(X_U, [0.4, 0.2, 0.7, 1, 0.5], "C")


def test_fuzzy_operations():
    np.testing.assert_allclose((~A).membership, [1, 0, 0.5, 0.7, 0.8])
    np.testing.assert_allclose((A | B).membership, [0, 1, 0.7, 0.3, 0.4])
    np.testing.assert_allclose((A & B).membership, [0, 0.5, 0.5, 0.2, 0.2])
    np.testing.assert_allclose((A - B).membership, [0, 0.5, 0.3, 0.3, 0.2])
    np.testing.assert_allclose((B - A).membership, [0, 0, 0.5, 0.2, 0.4])


def test_fuzzy_properties():
    props = check_properties(A, B, C)
    for name in props:
        if "excluded middle" in name or "non-contradiction" in name:
            assert not props[name], name  # these two laws FAIL for fuzzy sets
        else:
            assert props[name], name


def test_fuzzy_rejects_bad_membership():
    with pytest.raises(ValueError):
        FuzzySet([1, 2], [0.5, 1.2])


# --------------------------------------------------------------------- BAM
A_PAT = [[1, 1, -1, -1, 1], [-1, 1, 1, -1, -1], [1, 1, -1, 1, -1]]
B_PAT = [[-1, -1, 1, 1, -1, 1, -1], [-1, -1, -1, -1, -1, -1, -1], [1, 1, 1, -1, -1, -1, -1]]


def test_bam_recalls_stored_pairs():
    bam = BAM.from_pairs(A_PAT, B_PAT)
    for a, b in zip(A_PAT, B_PAT):
        np.testing.assert_array_equal(bam.forward(a), b)
        np.testing.assert_array_equal(bam.backward(b), a)


def test_bam_energy_never_increases():
    bam = BAM.from_pairs(A_PAT, B_PAT)
    trace = bam.recall([1, -1, -1, -1, 1])  # noisy version of a1
    assert all(e2 <= e1 + 1e-12 for e1, e2 in zip(trace.energies, trace.energies[1:]))


def test_bipolar_threshold_keeps_previous_state_at_zero():
    np.testing.assert_array_equal(bipolar_threshold([0, 2, -1], previous=[1, -1, 1]), [1, 1, -1])
    np.testing.assert_array_equal(bipolar_threshold([0, 2, -1]), [-1, 1, -1])
