"""
Unit tests for the analytical Lb-LSTM Jacobian and the projected Lyapunov adaptation law.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptation.jacobian_engine import (
    LSTMWeightLayout,
    LyapunovAdaptationLaw,
    lstm_forward,
    phi_jacobian,
    phi_jacobian_transpose_vec,
    smooth_projection,
)


@pytest.fixture
def setup() -> tuple[LSTMWeightLayout, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n, m, L = 2, 1, 5
    layout = LSTMWeightLayout(zeta_dim=2 * n + m + L + 1, hidden_dim=L, output_dim=n)
    theta = rng.normal(0.0, 0.5, layout.n_params)
    zeta = np.concatenate((rng.normal(size=2 * n + m + L), [1.0]))
    c_hat = rng.normal(size=L)
    return layout, theta, zeta, c_hat


def test_layout_views_are_column_major_vec(setup) -> None:
    layout, theta, _, _ = setup
    W_gates, W_h = layout.unpack(theta)
    d, L = layout.zeta_dim, layout.hidden_dim
    for k, name in enumerate(("c", "i", "f", "o")):
        W = layout.weight_matrix(theta, name)
        assert np.array_equal(W, W_gates[:, k * L:(k + 1) * L])
        # vec(W) stacks columns: vec(W)[j*d + r] = W[r, j]
        assert theta[layout.block_slice(name)][2 * d + 3] == W[3, 2]
    assert np.array_equal(layout.weight_matrix(theta, "h"), W_h)
    assert np.shares_memory(W_h, theta)


def test_jacobian_matches_central_finite_differences(setup) -> None:
    layout, theta, zeta, c_hat = setup
    J = phi_jacobian(lstm_forward(theta, zeta, c_hat, layout), theta, layout)
    assert J.shape == (layout.output_dim, layout.n_params)

    h = 1e-6
    J_num = np.zeros_like(J)
    for k in range(layout.n_params):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        J_num[:, k] = (lstm_forward(tp, zeta, c_hat, layout).phi - lstm_forward(tm, zeta, c_hat, layout).phi) / (2 * h)

    assert np.allclose(J, J_num, atol=1e-8, rtol=1e-6)
    for name in ("c", "i", "f", "o", "h"):
        assert np.abs(J[:, layout.block_slice(name)]).max() > 1e-3, f"block {name} is degenerate"


def test_fast_transpose_product_matches_explicit_jacobian(setup) -> None:
    layout, theta, zeta, c_hat = setup
    cache = lstm_forward(theta, zeta, c_hat, layout)
    e = np.array([0.3, -1.2])
    explicit = phi_jacobian(cache, theta, layout).T @ e
    fast = phi_jacobian_transpose_vec(cache, theta, layout, e)
    assert np.allclose(explicit, fast, atol=1e-13)


def test_projection_is_identity_inside_inner_ball() -> None:
    rng = np.random.default_rng(0)
    theta = rng.normal(size=50)
    theta *= 5.0 / np.linalg.norm(theta)
    tau = rng.normal(size=50)
    gamma = np.full(50, 2.0)
    assert np.array_equal(smooth_projection(theta, tau, gamma, w_bar=10.0, eps=0.1), tau)


def test_projection_removes_outward_component_on_boundary() -> None:
    rng = np.random.default_rng(1)
    gamma = rng.uniform(0.5, 3.0, 40)
    theta = rng.normal(size=40)
    theta *= 10.0 / np.linalg.norm(theta)
    outward = gamma * theta + rng.normal(size=40)  # theta^T tau > 0
    proj = smooth_projection(theta, outward, gamma, w_bar=10.0, eps=0.1)
    assert abs(float(theta @ proj)) < 1e-9 * np.linalg.norm(outward)
    inward = -outward
    assert np.array_equal(smooth_projection(theta, inward, gamma, w_bar=10.0, eps=0.1), inward)


def test_adaptation_law_keeps_weights_in_ball_under_persistent_push(setup) -> None:
    layout, _, zeta, c_hat = setup
    law = LyapunovAdaptationLaw(layout, gamma_gates=50.0, gamma_out=500.0, w_bar=6.0, proj_eps=0.1)
    theta = np.random.default_rng(3).normal(0.0, 0.3, layout.n_params)
    law.enforce_bound(theta)
    e = np.array([1.0, -1.0])
    max_norm = 0.0
    for _ in range(5000):
        cache = lstm_forward(theta, zeta, c_hat, layout)
        theta += 1e-3 * law.theta_dot(theta, cache, e)
        law.enforce_bound(theta)
        max_norm = max(max_norm, float(np.linalg.norm(theta)))
    assert max_norm <= 6.0 + 1e-12
    assert max_norm > 5.5  # the push actually reached the boundary layer
