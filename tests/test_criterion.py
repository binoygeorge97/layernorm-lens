"""Checks of the R6 criterion stage on synthetic arrays only (criterion_lib.py and
criterion.run_stages). No checkpoint, no observation and no criterion quantity of
any real model is computed."""

import copy
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
import criterion as crit  # noqa: E402
import criterion_lib as cl  # noqa: E402
import layouts  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(R6, "config.yaml"), encoding="utf-8"))
CC = CFG["criterion"]
geo = cl.geo
JUST = 1e-6  # relative offset that puts a synthetic statistic just on one side of a threshold


def test_config_matches_the_tagged_numbers():
    assert CC["kept_rtol"] == 1e-10 and CC["inside_quantile"] == 0.95
    assert CC["populated"] == dict(n_sub=5000, rng_seed=0, percentile=95)
    assert CC["sharp_min"] == 5.0 and CC["rho_levels"] == [1, 10, 100, 1000]
    assert CC["identification"]["return_ratio"] == 0.9
    assert CC["identification"]["e_over_e0_max"] == 0.1
    assert CC["tie_ratio"] == 1.1 and CC["borderline_band"] == 0.01
    assert CC["corollary1"] == dict(rtol=1e-10, n_state_lines=10, rng_seed=0)
    assert CC["g1"] == dict(tasks=["cartpole-swingup", "cheetah-run", "walker-run", "humanoid-run"],
                            min_pass=2)
    assert CC["susceptibility"] == dict(n_grid=2001, t_range_r_eff=3.0)
    assert [(c["task"], c["seed"]) for c in CC["calibration"]] == [
        ("cheetah-run", 1), ("walker-run", 1), ("humanoid-run", 1), ("cartpole-swingup", 2)]
    assert CC["tdmpc2_init"] == "tdmpc2/common/init.py"
    assert CFG["consistency"]["candidates"] == ["identity", "symlog", "layernorm"]


# --------------------------------------------------------------------------- #
# inputs: D7 (3) order                                                          #
# --------------------------------------------------------------------------- #


def test_episode_order():
    s, e = np.repeat(np.arange(5), 10), np.tile(np.arange(10), 5)
    assert cl.episode_order_ok(s, e, [0, 1, 2, 3, 4], 10)
    e2 = e.copy()
    e2[[3, 4]] = e2[[4, 3]]
    assert not cl.episode_order_ok(s, e2, [0, 1, 2, 3, 4], 10)
    s2 = s.copy()
    s2[[9, 10]] = s2[[10, 9]]
    assert not cl.episode_order_ok(s2, e, [0, 1, 2, 3, 4], 10)
    assert cl.episode_order_ok(s, None, [0, 1, 2, 3, 4], 10)  # D4 files: env_seed only
    assert not cl.episode_order_ok(s[::-1], None, [0, 1, 2, 3, 4], 10)


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


def test_g1_tasks_are_the_four_non_dog_tasks():
    """r6.md, Gate G1: "at least 2 of the 4 tasks other than dog-run"."""
    assert len(CC["g1"]["tasks"]) == 4
    assert set(CC["g1"]["tasks"]) == {"cartpole-swingup", "cheetah-run", "walker-run", "humanoid-run"}
    assert "dog-run" not in CC["g1"]["tasks"] and CC["g1"]["min_pass"] == 2


@pytest.mark.parametrize("rank", [None, 4])
def test_whitened_distance_equals_the_c_plus_form(rank):
    """r6.md: w(x, y) = √((x − y)ᵀ C⁺ (x − y)), with C⁺ the pseudo-inverse of C restricted
    to the kept eigenvectors. C⁺ is rebuilt here directly from C, independently of
    data_stats, and with rank=4 one eigenvalue is below the 1e-10 threshold (dropped)."""
    X = _data(rank=rank)
    S = cl.data_stats(X)
    C = np.cov(X, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    keep = w >= 1e-10 * w.max()
    C_plus = (V[:, keep] / w[keep]) @ V[:, keep].T
    assert keep.sum() == (5 if rank is None else 4) == S.m
    rng = np.random.default_rng(11)
    for _ in range(50):
        x, y = X[rng.integers(len(X))], X[rng.integers(len(X))] + rng.standard_normal(5)
        want = np.sqrt((x - y) @ C_plus @ (x - y))
        assert cl.whitened(S, x, y) == pytest.approx(want, rel=1e-10)
    assert np.allclose(S.C_plus, C_plus, rtol=1e-10, atol=1e-12)


def test_d2_degenerate_below_the_threshold():
    X = np.random.default_rng(0).standard_normal((500, 1)) * np.array([[1.0, 0.0]])
    X[:, 1] = 2.0
    S = cl.data_stats(X)
    assert S.m == 1 and not S.d2_kept


def test_canonical_sign_d7_7():
    assert np.array_equal(cl.canonical_sign([0.2, -0.9, 0.1]), [-0.2, 0.9, -0.1])
    assert np.array_equal(cl.canonical_sign([0.2, 0.9, -0.1]), [0.2, 0.9, -0.1])
    # a tie in magnitude is broken by the lowest index: index 0 (negative) decides
    assert np.array_equal(cl.canonical_sign([-1.0, 1.0, 0.5]), [1.0, -1.0, -0.5])
    S = cl.data_stats(_data())
    for i in range(S.k):
        v = S.eigvecs[:, i]
        assert v[np.argmax(np.abs(v))] > 0


def test_half_width_D_and_sign_invariance():
    X = _data()
    S = cl.data_stats(X)
    p = (X - S.mu) @ S.d1
    want = (np.percentile(p, 97.5) - np.percentile(p, 2.5)) / 2
    assert cl.half_width_D(S, X) == pytest.approx(want, rel=0, abs=0)
    p2 = (X - S.mu) @ (-S.d1)
    assert (np.percentile(p2, 97.5) - np.percentile(p2, 2.5)) / 2 == pytest.approx(want, rel=1e-14)


# --------------------------------------------------------------------------- #
# r6.md Criterion: each side of each threshold                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("side, expect", [(-1, True), (+1, False)])
def test_inside_each_side_of_the_chi2_quantile(side, expect):
    from scipy.stats import chi2
    X = _data()
    S = cl.data_stats(X)
    q = chi2.ppf(0.95, S.m)
    a = np.sqrt(q) * (1 + side * JUST)  # w(z*, mu)^2 = a^2 along d1
    z = S.mu + a * np.sqrt(S.eigvals[0]) * S.d1
    r = cl.inside(S, z)
    assert r["inside_w2"] == pytest.approx(a ** 2, rel=1e-10)
    assert r["inside_chi2_q"] == pytest.approx(q) and r["inside"] is expect


def test_inside_ignores_the_part_outside_the_kept_dimensions():
    X = _data(rank=4)
    S = cl.data_stats(X)
    z = S.mu + 100.0 * np.eye(5)[-1]  # along the dropped (constant) coordinate
    r = cl.inside(S, z)
    assert r["inside"] and r["inside_m"] == 4 and r["inside_norm_outside_kept"] == pytest.approx(100.0)


def test_nearest_other_matches_a_double_loop_and_counts_duplicates():
    Y = np.random.default_rng(3).standard_normal((300, 4))
    Y[7] = Y[200]  # D7 (3): an exact duplicate at another index counts, at 0
    nn = cl.nearest_other(Y, chunk=64)
    want = np.array([min(np.linalg.norm(Y[i] - Y[j]) for j in range(len(Y)) if j != i)
                     for i in range(len(Y))])
    assert np.allclose(nn, want, rtol=0, atol=1e-13) and nn[7] == 0 and nn[200] == 0


def test_populated_subsample_and_rule():
    X = _data(n=4000)
    S = cl.data_stats(X)
    r, idx, nn = cl.populated(S, X, S.mu, n_sub=1000, rng_seed=0)
    assert np.array_equal(idx, np.random.default_rng(0).choice(4000, 1000, replace=False))
    Y = (X[idx] - S.mu) @ S.whitener
    assert np.array_equal(nn, cl.nearest_other(Y))
    assert r["populated_threshold"] == pytest.approx(np.percentile(nn, 95))
    # D7 (3): z*'s nearest neighbour is taken from the subsample, not from all n rows
    assert r["populated_dist"] == pytest.approx(np.min(np.linalg.norm(Y, axis=1)))
    assert r["populated"]
    r2, _, _ = cl.populated(S, X, S.mu + 50 * np.sqrt(S.eigvals[0]) * S.d1, n_sub=1000)
    assert not r2["populated"]


def test_populated_nearest_neighbour_from_the_subsample_only():
    X = _data(n=4000)
    S = cl.data_stats(X)
    idx = np.random.default_rng(0).choice(4000, 1000, replace=False)
    outside = np.setdiff1d(np.arange(4000), idx)[0]
    r, _, _ = cl.populated(S, X, X[outside], n_sub=1000)
    assert r["populated_dist"] > 0  # z* on a row outside the subsample is not at distance 0
    r, _, _ = cl.populated(S, X, X[idx[5]], n_sub=1000)
    assert r["populated_dist"] < 1e-12 and r["populated_nearest_index"] == idx[5]  # 0 up to round-off


@pytest.mark.parametrize("side, expect", [(-1, True), (+1, False)])
def test_populated_each_side_of_the_95th_percentile(side, expect):
    # 1-D grid 0, 1, ..., 99 with the whole set as the subsample: every nearest-other
    # distance is 1/sigma, so the 95th percentile is 1/sigma; z* at raw distance
    # 1 -/+ JUST below the grid's first point
    X = np.arange(100.0)[:, None]
    S = cl.data_stats(X)
    z = np.array([-(1 + side * JUST)])
    r, _, _ = cl.populated(S, X, z, n_sub=100, rng_seed=0)
    sigma = np.sqrt(S.eigvals[0])
    assert r["populated_threshold"] == pytest.approx(1 / sigma, rel=1e-12)
    assert r["populated_dist"] == pytest.approx((1 + side * JUST) / sigma, rel=1e-12)
    assert r["populated"] is expect


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


@pytest.mark.parametrize("side, expect", [(+1, True), (-1, False)])
def test_sharp_each_side_of_5(side, expect):
    E, b = _layer()
    L = geo.lens(E, b, 1e-5)
    X = _data()
    S = cl.data_stats(X)
    r = cl.sharp(S, X, L)
    s = 5.0 * r["r_eff_d1"] / r["D"] * (1 + side * JUST)  # scales D, keeps d1 and r_eff(d1)
    Xs = S.mu + s * (X - S.mu)
    r2 = cl.sharp(cl.data_stats(Xs), Xs, L)
    assert r2["sharp_ratio"] == pytest.approx(5.0 * (1 + side * JUST), rel=1e-9)
    assert r2["sharp"] is expect


def test_rho_fractions():
    E, b = _layer()
    L = geo.lens(E, b, 1e-5)
    X = _data()
    r = cl.rho_fractions(L, X)
    rho = geo.lens_distance(L, X)
    assert r["frac_rho_le_10"] == pytest.approx(np.mean(rho <= 10))
    assert r["frac_rho_eff_le_10"] == pytest.approx(np.mean(rho / (1 + L.kappa) <= 10))


# --------------------------------------------------------------------------- #
# D6 (b), D7 (1), (2), G1                                                       #
# --------------------------------------------------------------------------- #


def _err(e_id, e_sl, e_ln, e0=10.0, e0_sl=None):
    return {"identity": dict(e=e_id, e0=e0), "symlog": dict(e=e_sl, e0=e0 if e0_sl is None else e0_sl),
            "layernorm": dict(e=e_ln, e0=e0)}


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


@pytest.mark.parametrize("side, expect", [(-1, True), (+1, False)])
def test_condition_ii_each_side_of_0p1(side, expect):
    e = 0.1 * (1 + side * JUST) * 10.0
    r = cl.condition_ii(_err(5.0, e, 6.0))
    assert r["lowest"] and r["e_over_e0"] == pytest.approx(0.1 * (1 + side * JUST))
    assert r["holds"] is expect


def test_condition_ii_uses_symlogs_own_e0():
    # D7 (2): e0 in the candidate's coordinates. Identity's e0 (10) would give 0.08.
    r = cl.condition_ii(_err(5.0, 0.8, 6.0, e0=10.0, e0_sl=4.0))
    assert r["e_over_e0"] == pytest.approx(0.2) and not r["holds"]


@pytest.mark.parametrize("frac, err, label", [
    (1.06, _err(9.0, 3.0, 9.5), "confirmed"),           # 0.1 threshold not applied
    (0.50, _err(9.0, 3.0, 9.5), "identified by consistency"),
    (1.06, _err(2.0, 3.0, 9.5), "unidentified"),
])
def test_humanoid_label(frac, err, label):
    _, r = cl.identification(1.0, cl.condition_ii(_err(1, 0.5, 2)), frac, cl.condition_ii(err))
    assert r["label"] == label


@pytest.mark.parametrize("side, within", [(-1, ["identity"]), (+1, [])])
def test_tie_rule_each_side_of_1p1(side, within):
    r = cl.tie_rule(_err(0.5 * 1.1 * (1 + side * JUST), 0.5, 2.0))
    assert r["ratios"]["identity"] == pytest.approx(1.1 * (1 + side * JUST))
    assert r["within"] == within and r["ambiguous"] is bool(within)


def test_tie_rule_every_reading_within_10_percent():
    r = cl.tie_rule(_err(0.52, 0.5, 0.54))
    assert r["within"] == ["identity", "layernorm"] and r["ambiguous"]
    c = cl.cartpole_seed1(True, r, {"symlog": True, "identity": True, "layernorm": False})
    assert c["required_readings"] == ["symlog", "identity", "layernorm"] and c["criterion"] is False
    c = cl.cartpole_seed1(True, r, {"symlog": True, "identity": True, "layernorm": True})
    assert c["criterion"] is True


def test_tie_rule_applies_only_when_symlog_is_lowest():
    r = cl.tie_rule(_err(0.49, 0.5, 2.0))
    assert not r["symlog_lowest"] and r["within"] == [] and not r["ambiguous"]
    assert r["ratios"]["identity"] == pytest.approx(0.98)  # still reported and scanned


TASKS = CC["g1"]["tasks"]


def _primary(**over):
    res = {(t, s): False for t in TASKS for s in (1, 2, 3)}
    res.update(over)
    return res


def test_tie_rule_changes_the_g1_vote():
    """cheetah-run passes; cartpole-swingup seed 1 passes under symlog but not under
    identity. Without a tie G1 passes (2 of 4); with identity within 10% of symlog,
    seed 1 needs identity too, fails, keeps the vote (seed 2's pass does not count),
    and G1 fails (1 of 4)."""
    primary = _primary()
    primary[("cheetah-run", 1)] = True
    primary[("cartpole-swingup", 1)] = True
    primary[("cartpole-swingup", 2)] = True
    by_reading = {"symlog": True, "identity": False, "layernorm": True}

    no_tie = cl.tie_rule(_err(0.5 * 1.5, 0.5, 2.0))
    cart = cl.cartpole_seed1(True, no_tie, by_reading)
    g = cl.g1_vote(cl.counting_checkpoints(TASKS, primary, cart, {}))
    assert cart["criterion"] and g["g1"] and g["n_pass"] == 2

    tie = cl.tie_rule(_err(0.5 * 1.05, 0.5, 2.0))
    cart = cl.cartpole_seed1(True, tie, by_reading)
    rows = cl.counting_checkpoints(TASKS, primary, cart, {})
    g = cl.g1_vote(rows)
    assert cart["ambiguous"] and cart["required_readings"] == ["symlog", "identity"]
    assert rows[0]["counting_seed"] == 1 and rows[0]["criterion"] is False
    assert not g["g1"] and g["n_pass"] == 1


def test_counting_checkpoints_fallback_and_flags():
    primary = _primary()
    primary[("cheetah-run", 1)] = primary[("cartpole-swingup", 2)] = True
    unident = cl.cartpole_seed1(False, cl.tie_rule(_err(1, 2, 3)), {})
    rows = cl.counting_checkpoints(TASKS, primary, unident, {})
    g = cl.g1_vote(rows)
    assert rows[0]["counting_seed"] == 2 and g["g1"] and g["n_pass"] == 2
    ident = cl.cartpole_seed1(True, cl.tie_rule(_err(5, 0.5, 6)), {"symlog": False})
    g = cl.g1_vote(cl.counting_checkpoints(TASKS, primary, ident, {}))
    assert not g["g1"] and g["rows"][0]["counting_seed"] == 1
    # D6 (a): a flagged counting checkpoint leaves G1, which still needs 2
    g = cl.g1_vote(cl.counting_checkpoints(TASKS, primary, unident, {("cheetah-run", 1): True}))
    assert not g["g1"] and g["n_counting"] == 3 and g["n_pass"] == 1
    # a flag on a seed that does not count changes nothing
    g = cl.g1_vote(cl.counting_checkpoints(TASKS, primary, unident, {("cheetah-run", 2): True}))
    assert g["g1"] and g["n_counting"] == 4


# --------------------------------------------------------------------------- #
# D7 (8) borderline scan                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rel, expect", [(0.0099, True), (-0.0099, True), (0.0101, False),
                                         (-0.0101, False), (0.01, True)])
def test_borderline_band(rel, expect):
    s = cl.borderline_scan([dict(where="x", statistic="s", value=5.0 * (1 + rel), threshold=5.0)])[0]
    assert s["borderline"] is expect and s["rel_distance"] == pytest.approx(abs(rel))


def test_borderline_items_cover_d7_8():
    row = dict(task="cheetah-run", seed=1, reading="identity", inside_w2=1.0, inside_chi2_q=11.07,
               populated_dist=0.2, populated_threshold=0.3, sharp_ratio=7.0, sharp_min=5.0)
    cii = cl.condition_ii(_err(5.0, 1.001, 6.0))   # e/e0 = 0.1001: borderline
    tie = cl.tie_rule(_err(1.1 * 1.0 * 1.005, 1.0, 9.0))  # identity ratio 1.1055: borderline
    items = cl.borderline_items([row, dict(row, reading="symlog")], cii, tie)
    assert len(items) == 2 * 3 + 1 + 2
    scan = cl.borderline_scan(items)
    hits = sorted(s["statistic"] for s in scan if s["borderline"])
    assert hits == ["D7 (1) ratio e_identity / e_symlog vs 1.1", "condition (ii) e/e0 vs 0.1"]


# --------------------------------------------------------------------------- #
# networks, Corollary 1 and D6 (g)                                              #
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


@pytest.mark.parametrize("eps", [1e-5, 1e-1])
def test_corollary1_on_lines_through_the_data(eps):
    """D7 (4): each line with its own s*, c_perp_l, kappa_l and r*_l."""
    E, b = _layer(k=4)
    L = geo.lens(E, b, eps)
    rng = np.random.default_rng(8)
    X = L.z_star + 3 * rng.standard_normal((200, 4))
    d = rng.standard_normal(4)
    d /= np.linalg.norm(d)
    x0 = X[17]
    r = cl.corollary1_data_line(L, E, b, x0, d, X, n_grid=401)
    assert r["max_rel_dev"] < 1e-12
    ln = geo.line(L, d, x0)
    assert abs((L.A @ d) @ ln["c_perp_l"]) < 1e-10 * np.linalg.norm(L.A @ d) * ln["norm_c_perp_l"]
    assert r["T"] >= 3 * r["r_eff_l"] and r["T"] >= np.max(np.abs((X - (x0 + r["s_star"] * d)) @ d)) - 1e-12
    # the check discriminates: the lens's own c_hat and kappa do not fit an off-centre line
    th = np.linspace(-1.2, 1.2, 101)
    pts = x0 + r["s_star"] * d + (r["r_star_l"] * np.tan(th))[:, None] * d
    qh = L.A @ d / np.linalg.norm(L.A @ d)
    wrong = np.sqrt(L.H) * (np.cos(th)[:, None] * L.c_hat + np.sin(th)[:, None] * qh) \
        / np.sqrt(1 + L.kappa * np.cos(th)[:, None] ** 2)
    hh = cl.ln_hat(E, b, eps, pts)
    assert np.max(np.linalg.norm(hh - wrong, axis=1) / np.linalg.norm(wrong, axis=1)) > 1e-3


def test_corollary1_data_line_through_z_star_is_the_g_line():
    E, b = _layer(k=4)
    L = geo.lens(E, b, 1e-5)
    d = np.eye(4)[1]
    X = L.z_star + np.random.default_rng(9).standard_normal((100, 4))
    r = cl.corollary1_data_line(L, E, b, L.z_star, d, X)
    t, info, dev = cl.corollary1_g_line(L, E, b, d, X)
    assert abs(r["s_star"]) < 1e-12 and r["r_star_l"] == pytest.approx(info["r_star"], rel=1e-12)
    assert r["T"] == pytest.approx(info["T"], rel=1e-12) and dev < 1e-12 and r["max_rel_dev"] < 1e-12


def test_data_line_rows():
    assert np.array_equal(cl.data_line_rows(25050), np.random.default_rng(0).choice(25050, 10, replace=False))


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
    t, info, dev = cl.corollary1_g_line(L, E, b, d, X, n_grid=201)
    assert dev < 1e-10
    o = cl.susceptibility(L, d, t, info, {"encoder": (enc, {"a0": (enc, X, d)})})["encoder"]
    h = 1e-6
    fd = (np.asarray(enc(jnp.asarray(L.z_star + h * d))) - np.asarray(enc(jnp.asarray(L.z_star - h * d)))) / (2 * h)
    assert o["g_prime_0"] == pytest.approx(np.linalg.norm(fd) * info["r_star"], rel=1e-6)
    assert o["S"] == pytest.approx(o["g_prime_0"] / info["r_star"])
    assert o["S_eff"] == pytest.approx(o["g_prime_0"] / info["r_eff"])
    near = np.abs(t) <= 3 * info["r_eff"]
    assert o["peak_J"] == pytest.approx(np.max(o["J_line"][near]))
    assert np.allclose(o["g_prime_line"], o["J_line"] * (info["r_star"] ** 2 + t ** 2) / info["r_star"])


# --------------------------------------------------------------------------- #
# input verification on synthetic files (criterion.planner_inputs, d4_inputs)   #
# --------------------------------------------------------------------------- #


def _planner_files(tmp_path, swap=False, bad_return=False):
    """A fake Drive with one planner data file (cartpole-swingup seed 2) and a manifest
    whose smoke row points at a file that does not exist (it must never be read)."""
    import json
    sys.path.insert(0, R6)
    import provenance as pv
    drive, res = tmp_path / "drive", tmp_path / "results"
    (drive / "data" / "r6" / "planner").mkdir(parents=True)
    res.mkdir()
    rng = np.random.default_rng(0)
    s, e = np.repeat(np.arange(5), 10), np.tile(np.arange(10), 5)
    if swap:
        e[[0, 1]] = e[[1, 0]]
    R = rng.uniform(0, 1, (50, 500)).astype(np.float32)
    f = drive / "data" / "r6" / "planner" / "cartpole-swingup-seed2.npz"
    np.savez(f, obs=rng.standard_normal((50, 501, 3)).astype(np.float32),
             actions=rng.uniform(-1, 1, (50, 500, 1)).astype(np.float32), rewards=R,
             env_seed=s, episode_in_seed=e)
    sha = pv.sha256(f)
    man = res / "planner_obs_manifest.csv"
    man.write_text("kind,file,bytes,sha256\n"
                   f"data,data/r6/planner/cartpole-swingup-seed2.npz,{f.stat().st_size},{sha}\n"
                   "smoke,data/r6/planner_smoke/never-there.npz,1,00\n", encoding="utf-8")
    ret = float(R.astype(np.float64).sum(1).mean()) * (1.5 if bad_return else 1.0)
    meta = dict(complete=True, smoke=False, kind="data", eval_mode=True, task="cartpole-swingup", seed=2,
                layout="public", planner_input="identity", data=dict(sha256=sha),
                returns=dict(episodes=50, return_mean=ret, fraction=1.001, flag_below_half=False))
    (res / "meta_cartpole-swingup-seed2.json").write_text(json.dumps(meta), encoding="utf-8")
    cfg = copy.deepcopy(CFG)
    cfg["criterion"].update(planner_manifest=str(man), planner_results=str(res))
    return cfg, str(drive), f


def test_planner_inputs_verify_and_ignore_smoke(tmp_path):
    cfg, drive, f = _planner_files(tmp_path)
    out, rec = crit.planner_inputs(cfg, drive, [("cartpole-swingup", 2, "public")])
    O, A, frac, flag = out[("cartpole-swingup", 2)]
    assert O.dtype == np.float64 and O.shape == (50, 501, 3) and frac == 1.001 and flag is False
    assert list(rec["files"]) == ["data/r6/planner/cartpole-swingup-seed2.npz"]


def test_planner_inputs_refuse_a_changed_file(tmp_path):
    cfg, drive, f = _planner_files(tmp_path)
    with open(f, "ab") as fh:
        fh.write(b"x")
    with pytest.raises(SystemExit):
        crit.planner_inputs(cfg, drive, [("cartpole-swingup", 2, "public")])


def test_planner_inputs_refuse_the_wrong_order(tmp_path):
    cfg, drive, _ = _planner_files(tmp_path, swap=True)
    with pytest.raises(SystemExit):
        crit.planner_inputs(cfg, drive, [("cartpole-swingup", 2, "public")])


def test_planner_inputs_refuse_returns_that_disagree_with_the_meta(tmp_path):
    cfg, drive, _ = _planner_files(tmp_path, bad_return=True)
    with pytest.raises(SystemExit):
        crit.planner_inputs(cfg, drive, [("cartpole-swingup", 2, "public")])


def test_d4_inputs(tmp_path):
    import json
    import provenance as pv
    drive = tmp_path / "drive"
    (drive / "data" / "r6").mkdir(parents=True)
    f = drive / "data" / "r6" / "cartpole-swingup-seed1.npz"
    np.savez(f, obs=np.zeros((50, 501, 3), np.float32), actions=np.zeros((50, 500, 1), np.float32),
             env_seed=np.repeat(np.arange(5), 10))
    man = tmp_path / "d4_obs_manifest.csv"
    man.write_text(f"file,bytes,sha256\ncartpole-swingup-seed1.npz,{f.stat().st_size},{pv.sha256(f)}\n",
                   encoding="utf-8")
    meta = tmp_path / "meta_d4.json"
    meta.write_text(json.dumps({"match": True}), encoding="utf-8")
    cfg = copy.deepcopy(CFG)
    cfg.update(tasks=["cartpole-swingup"], seeds=[1])
    cfg["criterion"].update(d4_manifest=str(man), d4_meta=str(meta))
    out, rec = crit.d4_inputs(cfg, str(drive), [("cartpole-swingup", 1)])
    assert out[("cartpole-swingup", 1)][0].dtype == np.float64
    meta.write_text(json.dumps({"match": False}), encoding="utf-8")
    with pytest.raises(SystemExit):
        crit.d4_inputs(cfg, str(drive), [("cartpole-swingup", 1)])


def test_tdmpc2_init_record_reads_the_commit(tmp_path):
    import hashlib
    import subprocess
    src = tmp_path / "tdmpc2_src"
    (src / "tdmpc2" / "common").mkdir(parents=True)
    body = b"import torch.nn as nn\n\ndef weight_init(m):\n    pass\n"
    (src / "tdmpc2" / "common" / "init.py").write_bytes(body)
    run = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "core.autocrlf=false"]
    for args in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "c"]):
        subprocess.run(run + args, cwd=src, check=True, capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()
    (src / "tdmpc2" / "common" / "init.py").write_bytes(b"changed in the working tree\n")
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["tdmpc2_src"] = str(src)
    cfg["source"]["repo_commit"] = commit
    r = crit.tdmpc2_init_record(cfg)
    assert r["commit"] == commit and r["path"] == "tdmpc2/common/init.py"
    assert r["sha256"] == hashlib.sha256(body).hexdigest()  # from the commit, not the working tree


# --------------------------------------------------------------------------- #
# the stage end to end, on synthetic checkpoints (criterion.run_stages)         #
# --------------------------------------------------------------------------- #


def _stage_cfg():
    cfg = copy.deepcopy(CFG)
    cfg["criterion"]["populated"]["n_sub"] = 100   # the synthetic sets have 4 x 51 = 204 rows
    cfg["criterion"]["susceptibility"]["n_grid"] = 101
    return cfg


def _synthetic_cks():
    """The six checkpoints run_stages needs: the two pre-release ones, the four D6 (c)
    calibration ones (which include the other G1 seeds), all with k = 4, a = 2 and 4
    episodes of 50 steps."""
    survey = {("cartpole-swingup", 1): "prerelease", ("cartpole-swingup", 2): "public",
              ("cheetah-run", 1): "public", ("walker-run", 1): "public",
              ("humanoid-run", 1): "public", ("humanoid-run", 3): "prerelease"}
    cks = {}
    for i, ((task, seed), layout) in enumerate(survey.items()):
        sd = _public_sd(seed=i) if layout == "public" else _prerelease_sd(seed=i)[0]
        rng = np.random.default_rng(100 + i)
        O = rng.standard_normal((4, 51, 4)) * np.array([2.0, 1.0, 0.5, 0.3]) + 0.3
        A = rng.uniform(-1, 1, (4, 50, 2))
        cks[(task, seed)] = crit.Checkpoint(task, seed, layout, sd, None, O, A, 1.0, False)
    return cks


def _quiet(*a, **k):
    pass


@pytest.fixture(scope="module")
def stage_result():
    return crit.run_stages(_stage_cfg(), _synthetic_cks(), log=_quiet)


def test_stage_runs_every_step_in_order(stage_result):
    res = stage_result
    assert res["status"] in (crit.STATUS_COMPLETE, crit.STATUS_PENDING)
    assert all(r["status"] in ("ok",) or "skipped" in r["status"] for r in res["cor1_rows"])
    # per checkpoint: mu along d1 and d2, 10 data states, 3 D6 (g) lines
    assert len([r for r in res["cor1_rows"] if r.get("max_rel_dev") is not None]) == 6 * (2 + 10 + 3)
    assert max(r["max_rel_dev"] for r in res["cor1_rows"] if "max_rel_dev" in r) < 1e-10
    # D7 (5): three readings for each pre-release checkpoint, symlog primary
    rows = res["crit_rows"]
    assert len(rows) == 4 + 2 * 3
    pre = [r for r in rows if r["layout"] == "prerelease"]
    assert sorted((r["task"], r["reading"], r["primary"]) for r in pre) == [
        ("cartpole-swingup", "identity", False), ("cartpole-swingup", "layernorm", False),
        ("cartpole-swingup", "symlog", True),
        ("humanoid-run", "identity", False), ("humanoid-run", "layernorm", False),
        ("humanoid-run", "symlog", True)]
    # the LayerNorm reading of a 4-vector lies in a 3-dimensional subspace: m = 3
    assert all(r["m"] == 3 for r in pre if r["reading"] == "layernorm")
    s2 = res["s2"]
    assert {(r["task"], r["seed"]) for r in s2["cons_rows"]} == set(_synthetic_cks())
    assert set(s2["tie"]["ratios"]) == {"identity", "layernorm"}
    g1 = res["g1"]
    assert [r["task"] for r in g1["rows"]] == TASKS
    assert len(res["borderline"]) == 3 * len(rows) + 1 + 2
    assert {r["direction"] for r in res["sus_rows"]} == {"d1", "d2", "u_min"}
    rec = [r for r in res["sus_rows"] if r.get("output") == "dynamics"]
    assert all(r["n_recorded"] == 4 * 50 and r["n_a0"] == 4 * 51 for r in rec)  # D7 (6)


def test_stage_writes_its_outputs(stage_result, tmp_path):
    crit.write_outputs(str(tmp_path), stage_result)
    names = {os.path.relpath(os.path.join(dp, f), tmp_path).replace(os.sep, "/")
             for dp, _, fs in os.walk(tmp_path) for f in fs}
    for f in ("corollary1.csv", "criterion.csv", "rho_fractions.csv", "consistency_planner.csv",
              "identification.json", "g1.json", "g1.csv", "susceptibility.csv", "borderline.json",
              "lines/cartpole-swingup-seed1.npz", "populated/cartpole-swingup-seed1-layernorm.npz"):
        assert f in names


def test_stage_stops_on_a_corollary1_failure(monkeypatch, tmp_path):
    real = cl.corollary1_data_line
    calls = []

    def one_bad(*a, **k):
        r = real(*a, **k)
        calls.append(1)
        if len(calls) == 3:
            r = dict(r, max_rel_dev=2e-10)
        return r

    monkeypatch.setattr(crit.cl, "corollary1_data_line", one_bad)
    res = crit.run_stages(_stage_cfg(), _synthetic_cks(), log=_quiet)
    assert res["status"] == crit.STATUS_STOPPED
    assert [r["status"] for r in res["cor1_rows"]].count("FAIL") == 1
    assert "s2" not in res and "crit_rows" not in res and "g1" not in res  # nothing after step 1
    crit.write_outputs(str(tmp_path), res)
    assert os.listdir(tmp_path) == ["corollary1.csv"]


def test_stage_marks_borderline_results_pending(stage_result):
    cfg = _stage_cfg()
    row = stage_result["crit_rows"][0]
    cfg["criterion"]["sharp_min"] = row["sharp_ratio"] * (1 + 0.005)  # within 1% of the threshold
    res = crit.run_stages(cfg, _synthetic_cks(), log=_quiet)
    assert res["status"] == crit.STATUS_PENDING
    hits = [b for b in res["borderline"] if b["borderline"]]
    assert any(b["statistic"].startswith("Sharp") and b["where"].startswith(f"{row['task']}-seed{row['seed']}")
               for b in hits)
    assert "g1" in res and "sus_rows" in res  # results are still written
