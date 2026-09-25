"""Checks of the D6 planner collector that need no checkpoints, torch or GPU
(planner_lib.py, provenance.py, and planner_collect.py's file handling). Everything
that needs real checkpoints is exercised on Colab through --smoke."""

import json
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R6 = os.path.join(ROOT, "experiments", "r6_tdmpc2")
sys.path.insert(0, R6)
import layouts  # noqa: E402
import planner_lib as pl  # noqa: E402
import provenance as pv  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(R6, "config.yaml")))
KEYS = json.load(open(os.path.join(ROOT, "results", "r6", "prerelease_keys.json")))
G = CFG["planner_collect"]["gate"]


def _has_tag(tag):
    return pv.git("cat-file", "-t", f"refs/tags/{tag}")[1] == "tag"


# --------------------------------------------------------------------------- #
# remap                                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ck", sorted(KEYS))
def test_remap_is_a_bijection_onto_the_public_layout(ck):
    shapes = {k: tuple(v) for k, v in KEYS[ck].items()}
    m = pl.remap_keys(shapes)
    assert set(m) == set(shapes) and len(set(m.values())) == len(m) == 60
    renamed = {m[k]: shapes[k] for k in shapes if not k.startswith(pl.Q_PREFIXES)}
    k_obs = shapes["_encoder.state.1.weight"][1]
    a = shapes["_pi.6.weight"][0] // 2
    assert renamed == pl.public_shapes(k_obs, a)  # names and shapes, exactly
    assert all(m[k] == k for k in shapes if k.startswith(pl.Q_PREFIXES))
    assert sum(k.startswith(pl.Q_PREFIXES) for k in shapes) == 20


def test_remap_refuses_unexpected_and_missing_keys():
    keys = list(KEYS["cartpole-swingup-1"])
    with pytest.raises(pl.RemapError, match="unexpected"):
        pl.remap_keys(keys + ["_encoder.state.3.weight"])
    with pytest.raises(pl.RemapError, match="missing"):
        pl.remap_keys([k for k in keys if k != "_dynamics.1.bias"])
    with pytest.raises(pl.RemapError, match="Q-ensemble"):
        pl.remap_keys([k for k in keys if k != "_target_Qs.params.9"])


def test_remap_prerelease_keeps_values():
    sd = {k: np.full(tuple(v), i, float) for i, (k, v) in enumerate(KEYS["humanoid-run-3"].items())}
    new, m = pl.remap_prerelease(sd)
    for old, newk in m.items():
        assert new[newk] is sd[old]


@pytest.mark.skipif(not _has_tag(pv.D6_TAG), reason="tag prereg-r6-d3 not fetched")
def test_remap_table_is_d6s():
    text = pv.git("show", f"{pv.D6_TAG}:{pv.D6_PATH}")[1]
    for row, _ in pl.PRERELEASE_REMAP:
        assert row in text, row
    assert len(pl.PREFIX_MAP) == 20  # 40 keys: .weight and .bias of each prefix


# --------------------------------------------------------------------------- #
# hashes                                                                        #
# --------------------------------------------------------------------------- #


def test_download_refuses_wrong_hash(tmp_path):
    src = tmp_path / "src.pt"
    src.write_bytes(b"released checkpoint")
    good = pv.sha256(src)
    url = src.as_uri()
    dst = tmp_path / "ck" / "dmcontrol" / "x-1.pt"
    with pytest.raises(pv.ProvenanceError, match="refusing"):
        pv.download_verified(url, str(dst), "0" * 64)
    assert not dst.exists() and not os.path.exists(str(dst) + ".part")
    assert pv.download_verified(url, str(dst), good) == good
    dst.write_bytes(b"tampered")  # an existing local file is checked too
    with pytest.raises(pv.ProvenanceError):
        pv.download_verified(url, str(dst), good)


# --------------------------------------------------------------------------- #
# symlog                                                                        #
# --------------------------------------------------------------------------- #


def test_symlog_is_the_lens_codes():
    eps0 = float(CFG["consistency"]["pos0_layernorm_eps"])
    o = np.random.default_rng(1).standard_normal((200, 7)).astype(np.float32) * 30
    o[0, :3] = [0.0, -0.0, 1e-8]
    ref = layouts.input_candidates(eps0)["symlog"](o.astype(np.float64))
    assert np.array_equal(pl.input_fn("symlog", eps0)(o.astype(np.float64)), ref)
    assert np.array_equal(ref, np.sign(o.astype(np.float64)) * np.log1p(np.abs(o.astype(np.float64))))
    p = pl.planner_input(o, "symlog", eps0)
    assert p.dtype == np.float32 and np.array_equal(p, ref.astype(np.float32))
    assert np.array_equal(pl.planner_input(o, "identity", eps0), o)


# --------------------------------------------------------------------------- #
# gate and controls on synthetic errors                                         #
# --------------------------------------------------------------------------- #


def _errs(median, mx, n=1001):
    e = np.full(n, median)
    e[: n // 2] = median / 10
    e[-1] = mx
    return e


def test_gate_rule_boundaries():
    assert pl.gate_rule(_errs(1e-5, 1e-3), G["median_max"], G["max_max"])["pass"]
    r = pl.gate_rule(_errs(1.01e-5, 1e-4), G["median_max"], G["max_max"])
    assert not r["pass"] and not r["median_ok"] and r["max_ok"]
    r = pl.gate_rule(_errs(1e-7, 1.01e-3), G["median_max"], G["max_max"])
    assert not r["pass"] and r["median_ok"] and not r["max_ok"]
    s = pl.summarise(np.arange(1, 101, dtype=float))
    assert s["median"] == 50.5 and s["max"] == 100 and s["n"] == 100


def test_evaluate_sets_needs_every_set():
    ok, bad = _errs(1e-7, 1e-4), _errs(1e-7, 5e-3)
    assert pl.evaluate_sets({"random": ok, "d4": ok}, G)["pass"]
    assert not pl.evaluate_sets({"random": ok, "d4": bad}, G)["pass"]


def test_control_outcomes_and_margins():
    ok = pl.evaluate_sets({"random": _errs(1e-7, 1e-4), "d4": _errs(1e-7, 1e-4)}, G)
    wide = pl.evaluate_sets({"random": _errs(0.3, 1.2), "d4": _errs(0.2, 2e-2)}, G)
    narrow = pl.evaluate_sets({"random": _errs(1e-3, 5e-3), "d4": _errs(1e-3, 5e-3)}, G)
    mixed = pl.evaluate_sets({"random": _errs(0.3, 1.2), "d4": _errs(1e-3, 9e-3)}, G)
    m = G["negative_min_max"]
    assert pl.control_outcomes(ok, wide, m)["as_stated"]
    c = pl.control_outcomes(ok, narrow, m)  # fails the rule, but not by > 1e-2
    assert c["positive_as_stated"] and not c["negative_as_stated"] and not c["as_stated"]
    assert not pl.control_outcomes(ok, mixed, m)["as_stated"]
    assert not pl.control_outcomes(wide, wide, m)["positive_as_stated"]
    assert not pl.control_outcomes(ok, ok, m)["negative_as_stated"]


def test_halting_rule_is_literal():
    ok = dict(positive_as_stated=True, negative_as_stated=True, as_stated=True)
    bad = dict(positive_as_stated=False, negative_as_stated=True, as_stated=False)
    gp, gf = {"pass": True}, {"pass": False}
    h = pl.halts(ok, {"a": gp, "b": gp})
    assert h["public_allowed"] and h["prerelease_allowed"] and not h["reasons"]
    h = pl.halts(ok, {"a": gp, "b": gf})  # gate failure: pre-release only
    assert h["public_allowed"] and not h["prerelease_allowed"] and "['b']" in h["reasons"][0]
    h = pl.halts(bad, None)  # control failure: everything
    assert not h["public_allowed"] and not h["prerelease_allowed"]
    assert any("positive control" in r for r in h["reasons"])
    assert not pl.halts(None, None)["public_allowed"]


# --------------------------------------------------------------------------- #
# states, lens distance, returns                                                #
# --------------------------------------------------------------------------- #


def test_random_states_use_ddof():
    X = np.random.default_rng(3).standard_normal((50, 11, 4)) * [1, 2, 3, 4] + [5, 0, -5, 1]
    R = pl.random_states(X, 1000, 0, 1)
    flat = X.reshape(-1, 4)
    z = np.random.default_rng(0).standard_normal((1000, 4))
    assert np.array_equal(R, z * flat.std(0, ddof=1) + flat.mean(0))
    assert not np.array_equal(R, pl.random_states(X, 1000, 0, 0))


def test_relative_errors():
    zl = np.array([[3.0, 4.0], [1.0, 0.0]])
    zp = zl + np.array([[0.0, 0.5], [0.0, 0.0]])
    assert np.allclose(pl.relative_errors(zp, zl), [0.1, 0.0])


def test_lens_distance_and_d1():
    rng = np.random.default_rng(4)
    E, b, eps = rng.standard_normal((32, 3)), rng.standard_normal(32), 1e-5
    geo = pl.lens_geometry()
    L = geo.lens(E, b, eps)
    X = rng.standard_normal((500, 3)) * [5.0, 1.0, 0.1]
    d = pl.d1(X)
    assert abs(abs(d[0]) - 1) < 1e-2 and np.isclose(np.linalg.norm(d), 1)
    r = geo.line(L, d)["r_eff"]
    dist, info = pl.lens_distance(np.stack([L.z_star, L.z_star + r * d]), E, b, eps, d)
    assert np.allclose(dist, [0.0, 1.0]) and np.isclose(info["r_eff_d1"], r)
    worst, _ = pl.worst_states(np.array([0.1, 0.5, 0.2]), np.stack([L.z_star] * 3), E, b, eps, d, 2)
    assert [w["index"] for w in worst] == [1, 2]


def test_return_summary_flag():
    assert pl.return_summary([400, 500], 1000, 0.5)["flag_below_half"]
    s = pl.return_summary([500, 500], 1000, 0.5)
    assert not s["flag_below_half"] and s["fraction"] == 0.5


# --------------------------------------------------------------------------- #
# planner_collect.py file handling                                              #
# --------------------------------------------------------------------------- #


def _cfg_with(tmp_path, match=True):
    import copy
    cfg = copy.deepcopy(CFG)
    drive = tmp_path / "drive"
    (drive / "data" / "r6").mkdir(parents=True)
    f = drive / "data" / "r6" / "cartpole-swingup-seed1.npz"
    np.savez(f, obs=np.zeros((2, 3, 5), np.float32))
    manifest = tmp_path / "manifest.csv"
    pv.write_manifest(manifest, [f], f.parent)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"match": match}))
    cfg["planner_collect"]["d4"].update(manifest=str(manifest), meta=str(meta))
    return cfg, drive, f


def test_verify_d4(tmp_path):
    import planner_collect as pc
    cfg, drive, f = _cfg_with(tmp_path)
    paths, info = pc.verify_d4(cfg, str(drive), [("cartpole-swingup", 1)])
    assert paths[("cartpole-swingup", 1)] == str(f) and info["meta_match"]
    assert pc.d4_obs(str(f)).shape == (6, 5)
    f.write_bytes(b"changed")
    with pytest.raises(SystemExit):
        pc.verify_d4(cfg, str(drive), [("cartpole-swingup", 1)])
    cfg, drive, f = _cfg_with(tmp_path / "b", match=False)
    with pytest.raises(SystemExit):
        pc.verify_d4(cfg, str(drive), [("cartpole-swingup", 1)])


def test_is_complete_and_smoke_never_counts(tmp_path, monkeypatch):
    import planner_collect as pc
    monkeypatch.setattr(pc, "ROOT", str(tmp_path / "repo"))
    paths = pc.Paths(CFG, str(tmp_path / "drive"), smoke=False)
    npz = os.path.join(paths.drive_data, "walker-run-seed2.npz")
    os.makedirs(paths.drive_data)
    open(npz, "wb").write(b"data")
    meta = dict(complete=True, smoke=False, returns=dict(episodes=50),
                data=dict(sha256=pv.sha256(npz)))
    mp = os.path.join(paths.res, "meta_walker-run-seed2.json")
    json.dump(meta, open(mp, "w"))
    assert pc.is_complete(paths, "walker-run", 2, CFG)
    json.dump(dict(meta, returns=dict(episodes=1)), open(mp, "w"))
    assert not pc.is_complete(paths, "walker-run", 2, CFG)
    json.dump(dict(meta, smoke=True), open(mp, "w"))
    assert not pc.is_complete(paths, "walker-run", 2, CFG)
    json.dump(dict(meta, data=dict(sha256="0" * 64)), open(mp, "w"))
    assert not pc.is_complete(paths, "walker-run", 2, CFG)
    smoke = pc.Paths(CFG, str(tmp_path / "drive"), smoke=True)
    assert smoke.data != paths.data and smoke.res != paths.res and smoke.kind == "smoke"
