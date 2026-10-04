"""Synthetic checks of the R6 exploratory analyses E1-E3 (experiments/r6_exploratory).
No real data or checkpoint is used."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "r6_exploratory"))
import run as ex  # noqa: E402

cl, geo = ex.cl, ex.geo


def _setup(seed=0, k=4, H=32, n=2000):
    rng = np.random.default_rng(seed)
    E, b = rng.standard_normal((H, k)) / np.sqrt(k), rng.standard_normal(H)
    L = geo.lens(E, b, 1e-5)
    X = L.z_star + rng.standard_normal((n, k)) * np.array([3.0, 1.0, 0.5, 0.2])[:k]
    return L, X, cl.data_stats(X)


def test_e1_directions_and_summary():
    L, X, S = _setup()
    rows = ex.e1_directions(L, X, S)
    assert len(rows) == L.k
    for r in rows:
        u = cl.canonical_sign(L.principal_dirs[:, r["i"]])
        p = (X - S.mu) @ u
        lo, hi = np.percentile(p, [2.5, 97.5])
        assert r["D"] == pytest.approx((hi - lo) / 2)
        assert r["r_eff"] == pytest.approx(np.sqrt(L.norm_c_perp ** 2 + L.H * L.eps) / L.sing[r["i"]], rel=1e-12)
        assert r["z_inside"] == bool(lo <= (L.z_star - S.mu) @ u <= hi)
    s = ex.e1_summary(rows, L)
    assert s["max_ratio"] == max(r["ratio"] for r in rows)
    assert s["umin_i"] == int(np.argmin(L.principal_widths)) == 0  # widths ascend with i
    assert s["n_dirs_z_inside"] == L.k  # the synthetic cloud is centred on z*


def test_e1_z_outside_along_a_direction():
    L, X, S = _setup()
    Xs = X + 50 * cl.canonical_sign(L.principal_dirs[:, 1])  # move the data far along u_1
    rows = ex.e1_directions(L, Xs, cl.data_stats(Xs))
    assert not rows[1]["z_inside"] and rows[0]["z_inside"]


def test_e2_where():
    L, X, S = _setup()
    r = ex.e2_where(S, X, S.mu)  # at the mean, every state is farther out
    assert r["frac_w_x_gt_w_z"] == pytest.approx(1.0)
    r = ex.e2_where(S, X, X[5])  # on a data state: nearest-neighbour distance 0, percentile 0
    assert r["z_nn_dist"] == pytest.approx(0.0, abs=1e-12) and r["z_nn_percentile"] == pytest.approx(0.0)
    far = S.mu + 100 * np.sqrt(S.eigvals[0]) * S.d1
    r = ex.e2_where(S, X, far)
    assert r["frac_w_x_gt_w_z"] == 0.0 and r["z_nn_percentile"] == 100.0


def test_trunc_normal_respects_absolute_bounds():
    x = ex.trunc_normal(np.random.default_rng(0), (20000,), std=1.0, a=-0.5, b=0.5)
    assert x.min() >= -0.5 and x.max() <= 0.5
    y = ex.trunc_normal(np.random.default_rng(0), (20000,), std=0.02, a=-2.0, b=2.0)
    assert np.std(y) == pytest.approx(0.02, rel=0.03)


def test_e3_init_lens_is_degenerate_at_the_origin():
    L, X, S = _setup(k=4, H=32)
    sings, nAd = ex.e3_init_draws(4, 32, 0.02, (-2.0, 2.0), 1e-5, 50, np.random.default_rng(0), S.d1)
    assert sings.shape == (50, 4) and nAd.shape == (50,)
    rng = np.random.default_rng(0)
    E0 = ex.trunc_normal(rng, (32, 4), 0.02, -2.0, 2.0)
    L0 = geo.lens(E0, np.zeros(32), 1e-5)
    assert L0.degenerate and np.allclose(L0.z_star, 0)
    assert np.allclose(L0.principal_widths_eff, np.sqrt(32 * 1e-5) / sings[0], rtol=1e-12)
    assert nAd[0] == pytest.approx(np.linalg.norm(L0.A @ S.d1), rel=1e-12)
    row = ex.e3_row(L, S, X, sings, nAd, 32, 1e-5)
    D = cl.half_width_D(S, X)
    assert row["init_r_eff_d1_over_D_p50"] == pytest.approx(np.median(np.sqrt(32e-5) / nAd / D))
    assert row["trained_r_eff_d1_over_D"] == pytest.approx(1 / cl.sharp(S, X, L)["sharp_ratio"])
    assert row["init_norm_z_star"] == 0.0 and row["init_dist_z_mu"] == pytest.approx(np.linalg.norm(S.mu))
