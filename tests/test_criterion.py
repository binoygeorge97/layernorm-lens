"""Checks of the R6 criterion stage on synthetic arrays only (criterion_lib.py). No
checkpoint, no observation and no criterion quantity of any real model is computed."""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R6 = os.path.join(ROOT, "experiments", "r6_tdmpc2")
sys.path.insert(0, R6)
import criterion_lib as cl  # noqa: E402
import layouts  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(R6, "config.yaml")))
CC = CFG["criterion"]
geo = cl.geo


def test_config_matches_the_tagged_numbers():
    assert CC["kept_rtol"] == 1e-10 and CC["inside_quantile"] == 0.95
    assert CC["populated"] == dict(n_sub=5000, rng_seed=0, percentile=95)
    assert CC["sharp_min"] == 5.0 and CC["rho_levels"] == [1, 10, 100, 1000]
    assert CC["identification"]["return_ratio"] == 0.9
    assert CC["identification"]["e_over_e0_max"] == 0.1
    assert CC["g1"] == dict(tasks=["cartpole-swingup", "cheetah-run", "walker-run", "humanoid-run"],
                            min_pass=2)
    assert CC["susceptibility"] == dict(n_grid=2001, t_range_r_eff=3.0, cor1_rtol=1e-10)
    assert [(c["task"], c["seed"]) for c in CC["calibration"]] == [
        ("cheetah-run", 1), ("walker-run", 1), ("humanoid-run", 1), ("cartpole-swingup", 2)]


# --------------------------------------------------------------------------- #
# definitions                                                                   #
# --------------------------------------------------------------------------- #


def _data(n=3000, k=5, seed=1, rank=None):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((k, k)) * np.array([3.0, 1.5, 1.0, 0.5, 0.2])[:k]
    X = rng.standard_normal((n, k)) @ B + rng.standard_normal(k)
    if rank is not None:  # a constant coordinate: one eigenvalue exactly 0
        X[:, -1] = 0.7
    return X


def test_data_stats_definitions():
    X = _data()
    S = cl.data_stats(X)
    C = np.cov(X, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    assert np.allclose(S.C, C, rtol=0, atol=0) and S.m == 5
    assert np.isclose(S.eigvals[0], w.max()) and abs(abs(S.d1 @ V[:, -1]) - 1) < 1e-12
    assert abs(abs(S.d2 @ V[:, -2]) - 1) < 1e-12 and S.d2_kept
    assert np.allclose(S.C_plus, np.linalg.inv(C), rtol=1e-10)
    x, y = X[0], X[1]
    assert np.isclose(cl.whitened(S, x, y), np.sqrt((x - y) @ np.linalg.inv(C) @ (x - y)), rtol=1e-12)


def test_kept_dimensions_drop_a_constant_coordinate():
    X = _data(rank=4)
    S = cl.data_stats(X)
    assert S.m == 4 and not S.kept[-1]
    assert np.allclose(S.C_plus, np.linalg.pinv(S.C, rcond=1e-10, hermitian=True), atol=1e-10)
    assert abs(S.d1[-1]) < 1e-12  # d1 is an eigenvector of the full C, already k-dimensional


def test_d2_degenerate_below_the_threshold():
    X = np.random.default_rng(0).standard_normal((500, 1)) * np.array([[1.0, 0.0]])
    X[:, 1] = 2.0
    S = cl.data_stats(X)
    assert S.m == 1 and not S.d2_kept


def test_half_width_D_and_sign_invariance():
    X = _data()
    S = cl.data_stats(X)
    p = (X - S.mu) @ S.d1
    want = (np.percentile(p, 97.5) - np.percentile(p, 2.5)) / 2
    assert cl.half_width_D(S, X) == pytest.approx(want, rel=0, abs=0)
    p2 = (X - S.mu) @ (-S.d1)
    assert (np.percentile(p2, 97.5) - np.percentile(p2, 2.5)) / 2 == pytest.approx(want, rel=1e-14)


def test_inside_against_chi2():
    from scipy.stats import chi2
    X = _data()
    S = cl.data_stats(X)
    z = S.mu + 0.5 * np.sqrt(S.eigvals[0]) * S.d1  # w^2 = 0.25
    r = cl.inside(S, z)
    assert r["inside_w2"] == pytest.approx(0.25, rel=1e-10)
    assert r["inside_chi2_q"] == pytest.approx(chi2.ppf(0.95, 5)) and r["inside"]
    far = S.mu + 10 * np.sqrt(S.eigvals[0]) * S.d1
    assert not cl.inside(S, far)["inside"]


def test_nearest_other_matches_a_double_loop_and_counts_duplicates():
    Y = np.random.default_rng(3).standard_normal((300, 4))
    Y[7] = Y[200]  # an exact duplicate at another index is the nearest other, at 0
    nn = cl.nearest_other(Y, chunk=64)
    want = np.array([min(np.linalg.norm(Y[i] - Y[j]) for j in range(len(Y)) if j != i)
                     for i in range(len(Y))])
    assert np.allclose(nn, want, rtol=0, atol=1e-13) and nn[7] == 0 and nn[200] == 0


def test_populated_subsample_and_rule():
    X = _data(n=4000)
    S = cl.data_stats(X)
    r, idx = cl.populated(S, X, S.mu, n_sub=1000, rng_seed=0)
    assert np.array_equal(idx, np.random.default_rng(0).choice(4000, 1000, replace=False))
    Y = (X[idx] - S.mu) @ S.whitener
    assert r["populated_threshold"] == pytest.approx(np.percentile(cl.nearest_other(Y), 95))
    assert r["populated_dist"] == pytest.approx(np.min(np.linalg.norm(Y, axis=1)))
    assert r["populated"]  # the mean of a Gaussian cloud is populated
    r2, _ = cl.populated(S, X, S.mu + 50 * np.sqrt(S.eigvals[0]) * S.d1, n_sub=1000)
    assert not r2["populated"]


def _layer(H=32, k=5, seed=0, scale_b=1.0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((H, k)) / np.sqrt(k), scale_b * rng.standard_normal(H)


def test_sharp_uses_r_eff_along_d1_through_z_star():
    E, b = _layer()
    L = geo.lens(E, b, 1e-5)
    X = _data()
    S = cl.data_stats(X)
    r = cl.sharp(S, X, L)
    q = np.linalg.norm(L.A @ S.d1)
    assert r["r_eff_d1"] == pytest.approx(np.sqrt(L.norm_c_perp ** 2 + L.H * 1e-5) / q, rel=1e-12)
    assert r["r_star_d1"] == pytest.approx(L.norm_c_perp / q, rel=1e-12)
    assert r["sharp"] == (r["D"] / r["r_eff_d1"] >= 5)


def test_rho_fractions():
    E, b = _layer()
    L = geo.lens(E, b, 1e-5)
    X = _data()
    r = cl.rho_fractions(L, X)
    rho = geo.lens_distance(L, X)
    assert r["frac_rho_le_10"] == pytest.approx(np.mean(rho <= 10))
    assert r["frac_rho_eff_le_10"] == pytest.approx(np.mean(rho / (1 + L.kappa) <= 10))


# --------------------------------------------------------------------------- #
# D6 (b) and G1                                                                 #
# --------------------------------------------------------------------------- #


def _err(e_id, e_sl, e_ln, e0=10.0):
    return {"identity": dict(e=e_id, e0=e0), "symlog": dict(e=e_sl, e0=e0), "layernorm": dict(e=e_ln, e0=e0)}


@pytest.mark.parametrize("frac_g, err_g, identified", [
    (0.95, _err(1.0, 0.5, 2.0), True),
    (0.85, _err(1.0, 0.5, 2.0), False),      # (i) fails
    (0.95, _err(0.4, 0.5, 2.0), False),      # symlog not lowest
    (0.95, _err(3.0, 2.0, 4.0), False),      # e/e0 = 0.2 > 0.1
    (0.90, _err(1.0, 1.0, 2.0), True),       # boundaries are inclusive
])
def test_cartpole_identification(frac_g, err_g, identified):
    g, _ = cl.identification(frac_g, cl.condition_ii(err_g), 1.0, cl.condition_ii(_err(1, 0.5, 2)))
    assert g["identified"] is identified


@pytest.mark.parametrize("frac, err, label", [
    (1.06, _err(9.0, 3.0, 9.5), "confirmed"),           # 0.1 threshold not applied
    (0.50, _err(9.0, 3.0, 9.5), "identified by consistency"),
    (1.06, _err(2.0, 3.0, 9.5), "unidentified"),
])
def test_humanoid_label(frac, err, label):
    _, r = cl.identification(1.0, cl.condition_ii(_err(1, 0.5, 2)), frac, cl.condition_ii(err))
    assert r["label"] == label


def test_g1_vote_fallback_and_flags():
    tasks = CC["g1"]["tasks"]
    res = {(t, s): False for t in tasks for s in (1, 2, 3)}
    res[("cheetah-run", 1)] = res[("cartpole-swingup", 2)] = True
    v = cl.g1_vote(res, cartpole_s1_identified=False, flags={}, tasks=tasks)
    assert v["g1"] and v["n_pass"] == 2 and v["rows"][0]["counting_seed"] == 2
    v = cl.g1_vote(res, cartpole_s1_identified=True, flags={}, tasks=tasks)
    assert not v["g1"] and v["rows"][0]["counting_seed"] == 1
    v = cl.g1_vote(res, False, {("cheetah-run", 1): True}, tasks)
    assert not v["g1"] and v["n_counting"] == 3 and v["n_pass"] == 1


# --------------------------------------------------------------------------- #
# networks and D6 (g)                                                           #
# --------------------------------------------------------------------------- #


def _normed(sd, p, fin, fout, rng):
    sd[f"{p}.weight"] = rng.standard_normal((fout, fin)) / np.sqrt(fin)
    sd[f"{p}.bias"] = rng.standard_normal(fout)
    sd[f"{p}.ln.weight"] = 1 + 0.1 * rng.standard_normal(fout)
    sd[f"{p}.ln.bias"] = 0.1 * rng.standard_normal(fout)


def _public_sd(k=4, a=2, H=24, lat=16, mlp=24, seed=0):
    rng = np.random.default_rng(seed)
    sd = {}
    _normed(sd, "_encoder.state.0", k, H, rng)
    _normed(sd, "_encoder.state.1", H, lat, rng)
    for i, (fi, fo) in enumerate([(lat + a, mlp), (mlp, mlp), (mlp, lat)]):
        _normed(sd, f"_dynamics.{i}", fi, fo, rng)
    _normed(sd, "_pi.0", lat, mlp, rng)
    _normed(sd, "_pi.1", mlp, mlp, rng)
    sd["_pi.2.weight"] = rng.standard_normal((2 * a, mlp))
    sd["_pi.2.bias"] = rng.standard_normal(2 * a)
    return sd


def _prerelease_sd(**kw):
    """The same parameters under the pre-release names (D6 (b) table, inverted)."""
    pub = _public_sd(**kw)
    names = {"_encoder.state.0": "_encoder.state.1", "_encoder.state.0.ln": "_encoder.state.2",
             "_encoder.state.1": "_encoder.state.4", "_encoder.state.1.ln": "_encoder.state.5",
             "_dynamics.0": "_dynamics.0.0", "_dynamics.0.ln": "_dynamics.0.1",
             "_dynamics.1": "_dynamics.0.3", "_dynamics.1.ln": "_dynamics.0.4",
             "_dynamics.2": "_dynamics.0.6", "_dynamics.2.ln": "_dynamics.1",
             "_pi.0": "_pi.0", "_pi.0.ln": "_pi.1", "_pi.1": "_pi.3", "_pi.1.ln": "_pi.4",
             "_pi.2": "_pi.6"}
    return {names[k.rsplit(".", 1)[0]] + "." + k.rsplit(".", 1)[1]: v for k, v in pub.items()}, pub


@pytest.mark.parametrize("layout", ["public", "prerelease"])
def test_jax_networks_match_layouts(layout):
    sd = _public_sd() if layout == "public" else _prerelease_sd()[0]
    enc, dyn, _ = layouts.build_networks(sd, layout, 1e-5, 8, None, input_fn=lambda o: o)
    ej = cl.jax_mlp(cl.mlp_spec(sd, layout, "_encoder.state", "simnorm"), 1e-5, 8)
    dj = cl.jax_mlp(cl.mlp_spec(sd, layout, "_dynamics", "simnorm"), 1e-5, 8)
    X = np.random.default_rng(1).standard_normal((50, 4)) * 3
    za = np.concatenate([enc(X), np.random.default_rng(2).uniform(-1, 1, (50, 2))], -1)
    assert np.max(np.abs(enc(X) - np.asarray(ej(jnp.asarray(X))))) < 1e-13
    assert np.max(np.abs(dyn(za) - np.asarray(dj(jnp.asarray(za))))) < 1e-13


def test_prerelease_and_public_forms_agree():
    pre, pub = _prerelease_sd()
    X = np.random.default_rng(4).standard_normal((20, 4))
    f_pre = cl.jax_mlp(cl.mlp_spec(pre, "prerelease", "_encoder.state", "simnorm"), 1e-5, 8)
    f_pub = cl.jax_mlp(cl.mlp_spec(pub, "public", "_encoder.state", "simnorm"), 1e-5, 8)
    assert np.array_equal(np.asarray(f_pre(jnp.asarray(X))), np.asarray(f_pub(jnp.asarray(X))))


@pytest.mark.parametrize("eps", [1e-5, 1e-1])
def test_corollary1_closed_form(eps):
    E, b = _layer(k=4)
    L = geo.lens(E, b, eps)
    d = np.random.default_rng(5).standard_normal(4)
    d /= np.linalg.norm(d)
    t = np.linspace(-20, 20, 401)
    hh = cl.ln_hat(E, b, eps, L.z_star + t[:, None] * d)
    cf = cl.corollary1(L, d, t)
    assert np.max(np.linalg.norm(hh - cf, axis=1) / np.linalg.norm(cf, axis=1)) < 1e-12


def test_theta_grid():
    E, b = _layer(k=4)
    L = geo.lens(E, b, 1e-5)
    d = np.eye(4)[0]
    X = L.z_star + np.random.default_rng(6).standard_normal((100, 4))
    t, info = cl.theta_grid(L, d, X)
    assert len(t) == 2001 and t[0] == pytest.approx(-info["T"]) and t[-1] == pytest.approx(info["T"])
    th = np.arctan(t / info["r_star"])
    assert np.allclose(np.diff(th), np.diff(th)[0], rtol=1e-9)
    assert info["T"] >= 3 * info["r_eff"] and info["T"] >= np.max(np.abs((X - L.z_star) @ d)) - 1e-12


def test_susceptibility_against_finite_differences():
    sd = _public_sd()
    enc = cl.jax_mlp(cl.mlp_spec(sd, "public", "_encoder.state", "simnorm"), 1e-5, 8)
    E, b = sd["_encoder.state.0.weight"], sd["_encoder.state.0.bias"]
    L = geo.lens(E, b, 1e-5)
    X = L.z_star + np.random.default_rng(7).standard_normal((300, 4))
    d = cl.data_stats(X).d1
    r = cl.susceptibility(L, E, b, d, X, {"encoder": (enc, {"a0": (enc, X, d)})}, n_grid=201)
    assert r["cor1_ok"]
    o = r["out"]["encoder"]
    h = 1e-6
    fd = (np.asarray(enc(jnp.asarray(L.z_star + h * d))) - np.asarray(enc(jnp.asarray(L.z_star - h * d)))) / (2 * h)
    assert o["g_prime_0"] == pytest.approx(np.linalg.norm(fd) * r["r_star"], rel=1e-6)
    assert o["S"] == pytest.approx(o["g_prime_0"] / r["r_star"])
    assert o["S_eff"] == pytest.approx(o["g_prime_0"] / r["r_eff"])
    near = np.abs(r["t"]) <= 3 * r["r_eff"]
    assert o["peak_J"] == pytest.approx(np.max(o["J_line"][near]))
    assert np.allclose(o["g_prime_line"], o["J_line"] * (r["r_star"] ** 2 + r["t"] ** 2) / r["r_star"])
