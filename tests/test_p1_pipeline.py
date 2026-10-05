"""P-I analysis, LQR, hover check, data and stages, on synthetic plants only.

No surrogate is trained on quadrotor data and no lens quantity is computed on a
surrogate of the quadrotor: the end-to-end tests use a synthetic linear plant with the
quadrotor's dimensions (12 states, 4 inputs)."""

import copy
import json
import os
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, "experiments", "p1_quadrotor")
sys.path.insert(0, ROOT)
sys.path.insert(0, P1)
import p1_data as pdata  # noqa: E402
import p1_hover as phover  # noqa: E402
from control import lqr  # noqa: E402
from lens import analysis as an  # noqa: E402
from lens import geometry as geo  # noqa: E402
from lens import models  # noqa: E402


def _load(name, path):
    """Import experiments/p1_quadrotor/run.py under a unique name (other experiments
    have a run.py too)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prun = _load("p1_run", os.path.join(P1, "run.py"))
CFG = yaml.safe_load(open(os.path.join(P1, "config.yaml"), encoding="utf-8"))


# --------------------------------------------------------------------------- #
# a synthetic linear plant with the quadrotor's dimensions                      #
# --------------------------------------------------------------------------- #


def _linear_plant(seed=0, dt=0.01):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((12, 12)) * 0.3 - np.eye(12)
    B = rng.standard_normal((12, 4))
    step = lambda x, u: x + dt * (jnp.asarray(A) @ x + jnp.asarray(B) @ u)  # Euler: y = A x + B u exactly
    plant = pdata.Plant(step=step, x0=np.zeros(12), u0=np.full(4, 0.5), u_lo=np.zeros(4), u_hi=np.ones(4), dt=dt)
    return plant, A, B


def _small_cfg():
    cfg = copy.deepcopy(CFG)
    cfg["sampling"].update(n_train=400, n_val=100, n_test=100)
    cfg["grid"].update(H=32, branch_width=16, seeds=[0])  # H > k = 16 (rank A = k; theory.md Remark 1b)
    cfg["training"].update(max_steps=400, eval_every=100, log_every=200, batch_size=64)
    cfg["long_budget"].update(seeds=[0], max_steps=400, log_every=100, snapshot_every=200)
    cfg["analysis"]["corollary1"].update(n_state_lines=3, n_grid=201)
    cfg["analysis"]["attenuation"].update(n_grid=401)
    return cfg


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    plant, A, B = _linear_plant()
    hx, hu = np.full(12, 1.0), np.full(4, 0.4)
    d = tmp_path_factory.mktemp("p1data")
    ds, rows = prun.stage_generate(_small_cfg(), plant, hx, hu, str(d / "data"), str(d / "manifest.csv"))
    return plant, A, B, hx, hu, ds, d


# --------------------------------------------------------------------------- #
# data                                                                          #
# --------------------------------------------------------------------------- #


def test_dataset_shapes_standardisation_and_jacobians(synthetic):
    plant, A, B, hx, hu, ds, _ = synthetic
    tr = ds["train"]
    assert tr["Z"].shape == (400, 16) and tr["Y"].shape == (400, 12) and tr["J_std"].shape == (400, 12, 16)
    assert np.allclose(tr["Z"].mean(0), 0, atol=1e-12) and np.allclose(tr["Z"].std(0), 1, atol=1e-12)
    sc = ds["scalers"]
    J_true = np.hstack([A, B])  # linear plant with Euler: y = A x + B u
    assert np.allclose(ds["test"]["J_std"][0], J_true * sc["sd_z"][None, :] / sc["sd_y"][:, None], atol=1e-12)
    assert np.allclose(ds["test"]["y"], ds["test"]["z"] @ J_true.T, atol=1e-12)
    assert np.all(ds["train"]["z"][:, 12:] >= 0) and np.all(ds["train"]["z"][:, 12:] <= 1)


def test_manifest_round_trip_and_tamper_refusal(synthetic, tmp_path):
    _, _, _, _, _, ds, d = synthetic
    back = pdata.load_dataset(str(d / "data"), str(d / "manifest.csv"))
    assert np.array_equal(back["test"]["J_std"], ds["test"]["J_std"])
    assert np.array_equal(back["trim"]["Z"], ds["trim"]["Z"])
    import shutil
    shutil.copytree(d / "data", tmp_path / "data")
    with open(tmp_path / "data" / "val.npz", "ab") as f:
        f.write(b"x")
    import provenance as pv
    with pytest.raises(pv.ProvenanceError):
        pdata.load_dataset(str(tmp_path / "data"), str(d / "manifest.csv"))


def test_quadrotor_sampling_box_only():
    """Sampling and targets of the quadrotor plant, without any surrogate."""
    plant, params = pdata.quadrotor_plant(CFG["plant"])
    hx, hu = pdata.quadrotor_half_widths(CFG["sampling"]["half_widths"], plant.u0)
    assert hx.shape == (12,) and hu.shape == (4,)
    assert hx[6] == pytest.approx(np.deg2rad(30)) and hx[8] == pytest.approx(np.pi)
    x, u = pdata.sample(plant, hx, hu, 1000, np.random.default_rng(0))
    assert np.all(np.abs(x - plant.x0) <= hx) and np.all(u >= 0) and np.all(u <= params.f_max)


# --------------------------------------------------------------------------- #
# analysis                                                                      #
# --------------------------------------------------------------------------- #


def _exact_linear_surrogate(J_std, H=32, seed=0):
    """A prenorm surrogate with its branch switched off (W2 = 0): y = Wo (E z + b) + bo,
    exactly linear; Wo = J_std E⁺ makes its Jacobian equal J_std everywhere."""
    spec = models.SurrogateSpec(arch="prenorm", n_blocks=1, H=H, k=16, n_out=12, init="torch_default",
                                branch_width=8)
    p = models.init_params(spec, seed)
    E = np.asarray(p["E"])
    p["W2_0"] = jnp.zeros_like(p["W2_0"])
    p["b2_0"] = jnp.zeros_like(p["b2_0"])
    p["Wo"] = jnp.asarray(J_std @ np.linalg.pinv(E))
    p["bo"] = jnp.asarray(-np.asarray(p["Wo"]) @ np.asarray(p["b"]))
    return spec, p


def test_surrogate_equal_to_the_truth_gives_zero_error(synthetic):
    plant, A, B, hx, hu, ds, _ = synthetic
    spec, p = _exact_linear_surrogate(ds["test"]["J_std"][0])
    Js = an.jacobians(spec, p, ds["test"]["Z"])
    err = an.jacobian_error(Js, ds["test"]["J_std"])
    assert np.max(err) < 1e-12
    out = prun.analyse_one(_small_cfg(), spec, p, ds)
    assert out["err_median"] < 1e-12 and out["nf_near"] < 1e-12 and out["corollary1_max_rel_dev"] < 1e-10
    trims = phover.sample_trims(plant, hx, 7, 0)
    r, trows = prun.hover_one(_small_cfg(), spec, p, ds, plant, hx, hu, trims, phover.trim_truth(plant, trims))
    assert r["rel_err_A"] < 1e-10 and r["rel_err_B"] < 1e-10 and r["rel_err_K"] < 1e-8
    assert not r["h1_fail"] and not r["no_stabilising_gain"]
    assert len(trows) == 7 and max(t["variation_err"] for t in trows) < 1e-10 and r["trims_rel_err_A_median"] < 1e-10
    assert r["sign_agree_A"] == 1.0 and r["sign_agree_B"] == 1.0 and r["stable"]
    assert r["spectral_abscissa"] == pytest.approx(r["true_closed_loop_abscissa"], rel=1e-8)


def test_jacobians_against_finite_differences():
    spec = models.SurrogateSpec(arch="normedlinear", n_blocks=3, H=16, k=16, n_out=12, init="torch_default")
    p = models.init_params(spec, 2)
    z = np.random.default_rng(1).standard_normal((3, 16))
    J = an.jacobians(spec, p, z)
    h = 1e-6
    f = lambda zz: np.asarray(models.forward(spec, p, jnp.asarray(zz)))
    Jfd = np.stack([(f(z[0] + h * e) - f(z[0] - h * e)) / (2 * h) for e in np.eye(16)], 1)
    assert np.allclose(J[0], Jfd, atol=1e-7)
    assert np.allclose(an.jacobian_error(J, J), 0) and np.allclose(an.jacobian_error(J, J, "fro"), 0)
    assert an.jacobian_error(2 * J, J)[0] == pytest.approx(1.0)


def test_near_far_split():
    dist = np.arange(100.0)
    err = 1000.0 - dist  # error falls with lens distance
    r = an.near_far(err, dist, 0.1, 0.5, "median")
    assert r["n_near"] == 10 and r["n_far"] == 50
    assert r["near"] == pytest.approx(np.median(1000 - np.arange(10))) and r["far"] == pytest.approx(np.median(1000 - np.arange(50, 100)))
    assert r["ratio"] == pytest.approx(r["near"] / r["far"]) and r["near_dist_max"] == 9 and r["far_dist_min"] == 50
    assert an.near_far(err, dist, stat="mean")["near"] == pytest.approx(np.mean(1000 - np.arange(10)))


def _random_lens(seed=0, H=32, k=16, bias=1.0):
    rng = np.random.default_rng(seed)
    E, b = rng.standard_normal((H, k)) / np.sqrt(k), bias * rng.standard_normal(H)
    return E, b, geo.lens(E, b, 1e-5)


def test_lens_distances_sharpness_and_directions():
    E, b, L = _random_lens()
    Z = np.random.default_rng(1).standard_normal((500, 16))
    rho, kind = an.lens_distances(L, Z)
    assert kind == "rho" and np.allclose(rho, geo.lens_distance(L, Z))
    _, _, L0 = _random_lens(bias=0.0)
    rho0, kind0 = an.lens_distances(L0, Z)
    assert kind0 == "rho_eff" and np.all(np.isfinite(rho0))
    d1, ratio = an.data_d1(Z)
    s = an.direction_sharpness(L, Z, d1)
    D, lo, hi = an.half_width(Z, d1)
    assert s["D"] == D and s["sharpness"] == pytest.approx(D / s["r_eff"])
    assert s["coverage"] == pytest.approx(2 / np.pi * np.arctan(D / s["r_star"]))
    assert ratio >= 1.0 and d1[np.argmax(np.abs(d1))] > 0
    um = an.u_min(L)
    assert geo.line(L, um)["r_star"] == pytest.approx(L.principal_widths.min(), rel=1e-10)
    s0 = an.direction_sharpness(L0, Z, d1)  # degenerate: coverage uses r_eff
    assert s0["r_star"] == 0.0 and 0 < s0["coverage"] < 1


def test_corollary1_line_on_a_random_layer():
    E, b, L = _random_lens(seed=3)
    X = L.z_star + np.random.default_rng(2).standard_normal((300, 16))
    d = np.random.default_rng(4).standard_normal(16)
    d /= np.linalg.norm(d)
    for x0 in (X.mean(0), X[7], L.z_star):
        assert an.corollary1_line(L, E, b, x0, d, X, n_grid=401)["max_rel_dev"] < 1e-12


def test_attenuation_on_constructed_stacks():
    spec = models.SurrogateSpec(arch="prenorm", n_blocks=2, H=32, k=16, n_out=12, init="torch_default",
                                branch_width=16)
    p = models.init_params(spec, 0)
    for j in range(2):  # branches off: h stays affine in z, both blocks see the same LayerNorm input
        p[f"W2_{j}"] = jnp.zeros_like(p[f"W2_{j}"])
        p[f"b2_{j}"] = jnp.zeros_like(p[f"b2_{j}"])
    E, b = (np.asarray(m) for m in models.first_layer(spec, p))
    L = geo.lens(E, b, spec.eps)
    X = L.z_star + np.random.default_rng(0).standard_normal((200, 16))
    d = an.u_min(L)
    at = an.attenuation(spec, p, L, d, X, n_grid=801)
    assert at["attenuation_blocks"][0] == 1.0
    assert at["attenuation_blocks"][1] == pytest.approx(1.0, rel=1e-10)  # identical profiles
    assert at["sharpness_out"] == pytest.approx(1.0, rel=1e-10)  # the output is linear in z: flat speed
    assert at["sharpness_blocks"][0] > 1.0 and at["attenuation_out"] < 1.0


# --------------------------------------------------------------------------- #
# LQR and the hover check                                                       #
# --------------------------------------------------------------------------- #


def test_lqr_known_cases():
    A, B = np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]])
    K = lqr.lqr_continuous(A, B, np.eye(2), np.eye(1))
    assert np.allclose(K, [[1.0, np.sqrt(3.0)]], atol=1e-12)  # double integrator, Q = I, R = 1
    assert lqr.spectral_abscissa(A - B @ K) < 0
    Ad, Bd = np.eye(2) + 0.1 * A, 0.1 * B
    Kd = lqr.lqr_discrete(Ad, Bd, np.eye(2), np.eye(1))
    assert lqr.spectral_radius(Ad - Bd @ Kd) < 1


def test_hover_check_signs_and_instability():
    plant, A, B = _linear_plant()
    A_t, B_t = phover.true_y_jacobians(plant)
    assert np.allclose(A_t, A, atol=1e-12) and np.allclose(B_t, B, atol=1e-12)
    Q, R = phover.bryson(np.ones(12), np.full(4, 0.4))
    r = phover.hover_check(A_t, B_t, A_t, B_t, Q, R)
    assert r["rel_err_A"] == 0 and r["sign_agree_B"] == 1.0 and r["stable"]
    r = phover.hover_check(A_t, -B_t, A_t, B_t, Q, R)  # every input sign wrong
    assert r["sign_agree_B"] == 0.0 and r["rel_err_B"] == pytest.approx(2.0) and not r["stable"]


def test_physical_jacobian_inverts_the_standardisation(synthetic):
    plant, A, B, hx, hu, ds, _ = synthetic
    A_s, B_s = phover.physical_jacobian(ds["test"]["J_std"][0], ds["scalers"], 12)
    assert np.allclose(A_s, A, atol=1e-12) and np.allclose(B_s, B, atol=1e-12)


def test_quadrotor_true_hover_jacobians_match_the_rk4_map():
    plant, params = pdata.quadrotor_plant(CFG["plant"])
    A_y, B_y = phover.true_y_jacobians(plant)
    from plants import quadrotor as q
    Ad, Bd = q.linearize_step(plant.x0, plant.u0, params, plant.dt)
    assert np.allclose(A_y, (np.asarray(Ad) - np.eye(12)) / plant.dt, atol=1e-10)
    assert np.allclose(B_y, np.asarray(Bd) / plant.dt, atol=1e-8)


# --------------------------------------------------------------------------- #
# stages                                                                        #
# --------------------------------------------------------------------------- #


def test_grid_is_the_planned_40():
    g = prun.grid(CFG)
    assert len(g) == 40 and len({n for n, _, _ in g}) == 40
    assert {(s.arch, s.init, s.n_blocks, s.H, s.eps) for _, s, _ in g} == {
        (a, i, nb, 128, 1e-5) for a in models.ARCHS for i in models.INITS for nb in (1, 3)}


def test_train_and_analyse_stages_on_the_synthetic_plant(synthetic, tmp_path):
    plant, A, B, hx, hu, ds, _ = synthetic
    cfg = _small_cfg()
    members = [m for m in prun.grid(cfg) if m[1].arch == "normedlinear" and m[1].n_blocks == 3][:2]
    rows = prun.stage_train(cfg, ds, str(tmp_path / "ck"), str(tmp_path / "res"), members)
    assert len(rows) == 2 and all(r["best_val"] < np.inf for r in rows)
    for name, spec, seed in members:
        assert (tmp_path / "res" / "history" / f"{name}.csv").exists()
        h = np.load(tmp_path / "res" / "history" / f"{name}.npz")
        assert list(h["step"]) == [0, 200, 400] and h["z_star"].shape == (3, 16)
        p = prun.load_params(str(tmp_path / "ck" / f"{name}.npz"))
        out = prun.analyse_one(cfg, spec, p, ds, seed)
        assert {"nf_ratio", "d1_sharpness", "u_min_coverage", "err_median", "z_star_to_mean", "affected_frac",
                "r_star_init_median", "attenuation_out", "u_min_attenuation_out"} <= set(out)
        assert 0.0 <= out["affected_frac"] <= 1.0
        if not out["degenerate"]:
            assert out["corollary1_max_rel_dev"] < 1e-10
        r, _ = prun.hover_one(cfg, spec, p, ds, plant, hx, hu)
        assert {"rel_err_A", "rel_err_B", "sign_agree_A", "h1_fail", "no_stabilising_gain"} <= set(r)
        assert "p4_ratio_u_min" in out and "p4_ratio_d1" in out


def test_benchmark_stage_runs_on_random_targets(tmp_path):
    cfg = _small_cfg()
    cfg["benchmark"].update(n_train=200, n_val=50)
    members = [m for m in prun.grid(cfg) if m[2] == 0][:2]
    rows, summary = prun.stage_benchmark(cfg, str(tmp_path), members, steps=200)
    assert len(rows) == 2 and all(r["seconds_per_step"] > 0 and r["lens_record_seconds"] > 0 for r in rows)
    assert (tmp_path / "benchmark.csv").exists() and summary["n_models"] == 2


@pytest.mark.parametrize("stage", prun.GATED)
def test_gated_stages_refuse_without_the_prereg_tag(stage):
    r = subprocess.run([sys.executable, os.path.join(P1, "run.py"), "--config", os.path.join(P1, "config.yaml"), stage],
                       capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode != 0 and "prereg-p1" in (r.stderr + r.stdout)


# --------------------------------------------------------------------------- #
# the revision queue: minibatch default, long-budget subset, array runs, P5     #
# --------------------------------------------------------------------------- #


def test_config_training_defaults_and_long_budget_subset():
    t = CFG["training"]
    assert (t["batch_size"], t["eval_every"], t["patience_frac"], t["tol"], t["max_steps"]) == (2048, 500, 0.1, 0.01,
                                                                                                100000)
    g = prun.grid(CFG, "long_budget")
    assert len(g) == 10 and {(s.arch, s.init, s.n_blocks, s.H) for _, s, _ in g} == {
        (a, "zero_bias", 1, 128) for a in models.ARCHS}
    tc = prun.train_cfg(CFG, 3, long=True)
    assert tc["patience_frac"] is None and tc["max_steps"] == CFG["long_budget"]["max_steps"] and tc["seed"] == 3
    assert CFG["long_budget"]["max_steps"] > t["max_steps"]


def test_paths_redirect_outputs_but_not_the_manifest(tmp_path):
    P = prun.paths(CFG, str(tmp_path))
    assert P["data"].startswith(str(tmp_path)) and P["ck"].startswith(str(tmp_path)) and P["res"].startswith(str(tmp_path))
    assert P["manifest"] == prun.paths(CFG)["manifest"] and P["manifest"].startswith(prun.ROOT)
    assert prun.paths(CFG, str(tmp_path), str(tmp_path / "d"))["data"] == str(tmp_path / "d")


@pytest.fixture(scope="module")
def long_run(synthetic, tmp_path_factory):
    _, _, _, _, _, ds, _ = synthetic
    cfg = _small_cfg()
    base = tmp_path_factory.mktemp("scratch")
    members = prun.grid(cfg, "long_budget")
    assert len(members) == 2  # 2 architectures x 1 seed
    rows = prun.stage_train(cfg, ds, str(base / "ck" / "long"), str(base / "res" / "train_long"), members[:1],
                            long=True, base=str(base), summary=False)
    return cfg, ds, base, members, rows


def test_long_run_snapshots_history_and_manifest(long_run):
    cfg, ds, base, members, rows = long_run
    name = members[0][0]
    assert rows[0]["stopped_step"] == 400 and not rows[0]["stopped_early"]
    assert sorted(os.listdir(base / "ck" / "long" / name)) == ["step0000000.npz", "step0000200.npz", "step0000400.npz"]
    last = prun.load_params(str(base / "ck" / "long" / f"{name}-last.npz"))
    snap = prun.load_params(str(base / "ck" / "long" / name / "step0000400.npz"))
    assert all(np.array_equal(last[k], snap[k]) for k in last)
    h = np.load(base / "res" / "train_long" / "history" / f"{name}.npz")
    assert list(h["step"]) == [0, 100, 200, 300, 400] and h["u_min"].shape == (5, 16)
    assert not (base / "res" / "train_long" / "train_summary.csv").exists()  # a single array task writes none
    import provenance as pv
    m = pv.read_manifest(str(base / "res" / "train_long" / "runs" / f"{name}_manifest.csv"))
    assert len(m) == 3 + 1 + 1 + 3
    rj = json.load(open(base / "res" / "train_long" / "runs" / f"{name}.json", encoding="utf-8"))
    assert rj["git_commit"] and "slurm" in rj and rj["run"]["name"] == name


def test_gather_verifies_every_run(long_run):
    cfg, ds, base, members, _ = long_run
    import provenance as pv
    rd = str(base / "res" / "train_long")
    with pytest.raises(pv.ProvenanceError, match="missing"):  # the second member never ran
        prun.stage_gather(rd, str(base), members, require_clean=False)
    dirty = json.load(open(base / "res" / "train_long" / "runs" / f"{members[0][0]}.json", encoding="utf-8"))["git_dirty"]
    if dirty:  # the tests ran on a working tree with uncommitted changes: gather must refuse by default
        with pytest.raises(pv.ProvenanceError, match="dirty tree"):
            prun.stage_gather(rd, str(base), members[:1])
    rows, mrows = prun.stage_gather(rd, str(base), members[:1], require_clean=False)
    assert len(rows) == 1 and len(mrows) == 8 and (base / "res" / "train_long" / "train_summary.csv").exists()
    snap = base / "ck" / "long" / members[0][0] / "step0000200.npz"
    good = snap.read_bytes()
    snap.write_bytes(good + b"x")
    try:
        with pytest.raises(pv.ProvenanceError, match="SHA-256 mismatch"):
            prun.stage_gather(rd, str(base), members[:1], require_clean=False)
    finally:
        snap.write_bytes(good)


def test_analyse_long_traces_prediction_1_over_snapshots(long_run):
    cfg, ds, base, members, _ = long_run
    rows = prun.stage_analyse_long(cfg, ds, str(base / "ck" / "long"), members[:1])
    assert [r["step"] for r in rows] == [0, 200, 400]
    assert rows[0]["degenerate"] and rows[0]["dist_kind"] == "rho_eff" and rows[0]["z_star_to_mean"] < 1e-12
    assert all(np.isfinite(r["nf_ratio"]) and r["u_min_r_eff_over_D"] > 0 for r in rows)


def test_p5_rule():
    steps = [0, 500, 1000, 1500]
    ok = prun.p5_run(steps, [0.008, 0.02, 0.09, 0.12], [np.inf, 0.5, 0.08, 0.05], 1500)
    assert ok["passes"] and ok["fold_end"] == pytest.approx(15.0) and ok["ratio_max"] == 0.12
    assert not prun.p5_run(steps, [0.008, 0.02, 0.09, 0.12], [np.inf, 0.5, 0.08, 0.05], 1000)["passes"]  # < 0.1
    assert not prun.p5_run(steps, [0.02, 0.05, 0.1, 0.12], [np.inf, 0.5, 0.08, 0.05], 1500)["passes"]  # fold 6
    assert not prun.p5_run(steps, [0.008, 0.02, 0.09, 0.12], [np.inf, 0.5, 0.08, 0.2], 1500)["passes"]  # kappa
    assert not prun.p5_run(steps, [0.008, 0.02, 0.09, 0.12], [np.inf] * 4, 1500)["passes"]  # still degenerate
    with pytest.raises(ValueError):
        prun.p5_run(steps, [0.008, 0.02, 0.09, 0.12], [np.inf] * 4, 750)  # the end step was not logged
    res = [dict(arch="a", passes=p) for p in (True, True, True, True, False)] + \
          [dict(arch="b", passes=p) for p in (True, True, True, False, False)]
    cells = prun.p5_cells(res, ("arch",), 4)
    assert [(c["arch"], c["n"], c["n_pass"], c["holds"]) for c in cells] == [("a", 5, 4, True), ("b", 5, 3, False)]


def test_stage_p5_reads_the_lens_logs(synthetic, tmp_path):
    _, _, _, _, _, ds, _ = synthetic
    cfg = _small_cfg()
    cfg["grid"].update(archs=["normedlinear"], n_blocks=[1], seeds=[0, 1])
    cfg["training"].update(log_every=100)  # every evaluation logged, as in the P-I config
    prun.stage_train(cfg, ds, str(tmp_path / "ck"), str(tmp_path / "res" / "train"))
    runs, cells = prun.stage_p5(cfg, str(tmp_path / "res"), "train")
    assert len(runs) == 4 and {r["init"] for r in runs} == {"zero_bias", "torch_default"}
    summ = {r["name"]: r for r in prun.read_csv(str(tmp_path / "res" / "train" / "train_summary.csv"))}
    assert all(r["end_step"] == int(summ[r["name"]]["best_step"]) for r in runs)
    assert [(c["arch"], c["n_blocks"], c["n"]) for c in cells] == [("normedlinear", 1, 2)]  # zero bias only


def test_p3_spearman_and_initial_r_star():
    rows = [dict(u_min_sharpness=s, d1_sharpness=s, u_min_coverage=s, d1_coverage=-s, affected_frac=a)
            for s, a in ((1, 0.1), (2, 0.3), (3, 0.2), (4, 0.4))]
    r = prun.p3_spearman(rows)
    assert r["u_min_sharpness"] == pytest.approx(0.8) and r["d1_coverage"] == pytest.approx(-0.8)
    # the PyTorch-default start lies inside the initial-lens check's range at H = 128, k = 16
    spec = models.SurrogateSpec(arch="prenorm", n_blocks=1, init="torch_default")
    assert 0.79 < prun.initial_r_star_median(spec, 0) < 1.07
    assert prun.initial_r_star_median(models.SurrogateSpec(init="zero_bias"), 0) < 0.05  # ε-limited


def test_init_lens_stage_from_initialisation_alone(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["init_lens"].update(n_seeds=6, n_inputs=2000)
    rows, summary = prun.stage_init_lens(cfg, str(tmp_path))
    assert len(rows) == 12 and (tmp_path / "init_lens_summary.csv").exists()
    zb = [r for r in rows if r["init"] == "zero_bias"]
    assert all(r["degenerate"] and r["norm_z_star"] == 0.0 for r in zb)
    assert all(r["r_eff_u_min"] <= r["r_eff_d1"] * (1 + 1e-12) for r in rows)  # u_min is the narrowest
    td = [r for r in rows if r["init"] == "torch_default"]
    assert all(not r["degenerate"] and 0.47 < r["r_star_init_median"] < 1.87 for r in td)
    q = {(s["init"], s["quantity"]): s for s in summary}
    assert q[("zero_bias", "r_eff_over_D_u_min")]["seed0"] == zb[0]["r_eff_over_D_u_min"]


def test_first_layer_is_shared_across_architectures_and_depths():
    for seed in range(3):
        Eb = [models.first_layer(s, models.init_params(s, seed)) for s in (
            models.SurrogateSpec(arch=a, n_blocks=nb, init="torch_default") for a in models.ARCHS for nb in (1, 3))]
        assert all(np.array_equal(Eb[0][0], e) and np.array_equal(Eb[0][1], b) for e, b in Eb)


# --------------------------------------------------------------------------- #
# the author's decisions of 5 Oct: P4's S ratio, H1, trims, the rules           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arch", models.ARCHS)
def test_truncated_trace_and_attenuation_S(arch):
    spec = models.SurrogateSpec(arch=arch, n_blocks=3, H=32, init="torch_default", branch_width=16)
    p = models.init_params(spec, 1)
    z = jnp.asarray(np.random.default_rng(0).standard_normal(16))
    assert np.array_equal(models.trace(spec, p, z, upto=3)[0], models.forward(spec, p, z))
    y1, hats = models.trace(spec, p, z, upto=1)
    spec1 = models.SurrogateSpec(arch=arch, n_blocks=1, H=32, init="torch_default", branch_width=16)
    if arch == "normedlinear":
        p1 = {k: p[k] for k in ("W0", "b0", "g0", "be0", "Wo", "bo")}
    else:
        p1 = {k: p[k] for k in ("E", "b", "g0", "be0", "W1_0", "b1_0", "W2_0", "b2_0", "Wo", "bo")}
    assert np.allclose(y1, models.forward(spec1, p1, z), rtol=0, atol=1e-14) and len(hats) == 1
    with pytest.raises(ValueError):
        models.trace(spec, p, z, upto=4)
    E, b = (np.asarray(m) for m in models.first_layer(spec, p))
    L = geo.lens(E, b, spec.eps)
    d = an.u_min(L)
    r = an.attenuation_S(spec, p, L, d)
    h = 1e-6
    fd = np.linalg.norm((np.asarray(models.forward(spec, p, jnp.asarray(L.z_star + h * d)))
                         - np.asarray(models.forward(spec, p, jnp.asarray(L.z_star - h * d)))) / (2 * h))
    assert r["S_out"] == pytest.approx(fd, rel=1e-6) and r["ratio"] == pytest.approx(r["S_out"] / r["S_block1"])
    r1 = an.attenuation_S(spec1, p1, L, d)
    assert r1["ratio"] == pytest.approx(1.0, rel=1e-14)


def test_no_stabilising_gain_counts_for_h1():
    A_t = np.array([[0.0, 1.0], [0.0, 0.0]])
    B_t = np.array([[0.0], [1.0]])
    Q, R = np.eye(2), np.eye(1)
    ok = phover.hover_check(A_t, B_t, A_t, B_t, Q, R)
    assert ok["lqr_ok"] and not ok["no_stabilising_gain"] and not ok["h1_fail"]
    # an unstable mode the input cannot reach: no stabilising LQR solution
    r = phover.hover_check(np.diag([1.0, 0.0]), np.array([[0.0], [1.0]]), A_t, B_t, Q, R)
    assert r["no_stabilising_gain"] and not r["stable"] and r["h1_fail"]
    # a sign flip alone is a failure too
    r = phover.hover_check(A_t, -B_t, A_t, B_t, Q, R)
    assert r["sign_agree_B"] < 1.0 and r["h1_fail"]


def test_quadrotor_trims_are_trims_inside_the_box():
    plant, _ = pdata.quadrotor_plant(CFG["plant"])
    hx, _ = pdata.quadrotor_half_widths(CFG["sampling"]["half_widths"], plant.u0)
    X = phover.sample_trims(plant, hx, 12, 3)
    inside, dev = phover.check_trims(plant, X, hx)
    assert inside and dev < 1e-12
    assert np.all(X[:, [6, 7, 9, 10, 11]] == 0.0) and np.all(np.abs(X - plant.x0) <= hx)
    A, B = phover.trim_truth(plant, X)
    A0, B0 = phover.true_y_jacobians(plant)
    # the truth depends on the trim only through yaw
    X2 = X.copy()
    X2[:, 8] = 0.0
    A2, B2 = phover.trim_truth(plant, X2)
    assert np.allclose(A2, A0[None], rtol=0, atol=1e-10) and np.allclose(B2, B0[None], rtol=0, atol=1e-10)
    assert np.allclose(A[0], phover.true_y_jacobians(plant, X[0], plant.u0)[0], rtol=0, atol=1e-12)


def test_trims_stage_regeneration_identity(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["hover"]["trims"].update(n=6)
    plant, _ = pdata.quadrotor_plant(cfg["plant"])
    hx, _ = pdata.quadrotor_half_widths(cfg["sampling"]["half_widths"], plant.u0)
    out = tmp_path / "trims"
    X, dev = prun.stage_trims(cfg, plant, hx, str(out))
    prun.stage_trims(cfg, plant, hx, str(out))  # a rerun reproduces the file byte for byte
    assert np.array_equal(prun.load_trims(cfg, str(tmp_path)), X)
    import provenance as pv
    cfg["hover"]["trims"]["seed"] += 1
    with pytest.raises(pv.ProvenanceError, match="differ"):
        prun.stage_trims(cfg, plant, hx, str(out))
    (out / "trims.csv").write_text((out / "trims.csv").read_text() + "\n")
    with pytest.raises(pv.ProvenanceError):
        prun.load_trims(cfg, str(tmp_path))


def _rows(vals):
    """analyse-like rows for the 40-model grid; vals(arch, init, nb, seed) -> dict."""
    return [dict(name=n, arch=s.arch, init=s.init, n_blocks=str(s.n_blocks), seed=str(seed),
                 **{k: str(v) for k, v in vals(s.arch, s.init, s.n_blocks, seed).items()})
            for n, s, seed in prun.grid(CFG)]


def test_prediction_rules_on_constructed_rows():
    import p1_rules as rules
    q = CFG["predictions"]

    def vals(arch, init, nb, seed):
        zero = init == "zero_bias"
        # P1 holds in the prenorm cells only; seed 4 of every cell fails clause (b)
        ratio = (5.0 if arch == "prenorm" else 2.0) if zero else 1.5
        ratio = 2.9 if seed == 4 and zero else ratio
        return dict(z_star_to_mean=0.05 if zero else 0.4, nf_ratio=ratio, r_star_init_median=0.94,
                    u_min_sharpness=10.0 * ratio + seed, affected_frac=ratio / 10.0 + seed / 1000.0,
                    p4_ratio_u_min=0.25 if nb == 3 else 1.0, h1_fail=zero and seed != 0,
                    no_stabilising_gain=False)
    rows = _rows(vals)
    p1 = rules.p1(rows, q)
    assert [(c["arch"], c["n_blocks"], c["n_pass"], c["holds"]) for c in p1] == [
        ("normedlinear", 1, 0, False), ("normedlinear", 3, 0, False), ("prenorm", 1, 4, True), ("prenorm", 3, 4, True)]
    g = rules.g2(p1, q)
    assert g["passes"] and g["p3_cells"] == [("prenorm", 1), ("prenorm", 3)]
    p2 = rules.p2(rows, q)
    assert p2["a_holds"] and p2["a_n"] == 20 and p2["holds"]
    p3 = rules.p3(rows, q)
    assert p3["holds"] and p3["rho"] > 0.7
    p4 = rules.p4(rows, q)
    assert p4["holds"] and len(p4["cells"]) == 4 and p4["point_consistent"] and p4["zero_bias_median"] == 0.25
    h1 = rules.h1(rows, q)
    assert h1["holds"] and all(c["n_pass"] == 4 for c in h1["cells"])
    p5 = rules.p5([dict(arch="normedlinear", holds="True"), dict(arch="prenorm", holds="False")], [])
    assert not p5["holds"]
    es = [dict(arch=r["arch"], init=r["init"], n_blocks=r["n_blocks"], seed=r["seed"], passes=str(r["arch"] == "prenorm"))
          for r in rows]
    race = rules.race(p1, es)
    assert race[2] == dict(arch="prenorm", n_blocks=1, p1_and_p5=4, p1_only=0, p5_only=1, neither=0)
    # a negative correlation fails prediction 3, whatever its size
    neg = [dict(r, affected_frac=str(-float(r["affected_frac"]))) for r in rows]
    assert not rules.p3(neg, q)["holds"]


def test_corollary1_deviation_stops_the_analysis():
    prun.check_corollary1(dict(corollary1_max_rel_dev=3e-14), 1e-10)
    prun.check_corollary1(dict(degenerate=True), 1e-10)  # degenerate: no Corollary 1 line, nothing to check
    for bad in (2e-10, float("nan")):
        with pytest.raises(RuntimeError, match="code error"):
            prun.check_corollary1(dict(corollary1_max_rel_dev=bad), 1e-10)
    assert CFG["analysis"]["corollary1"]["tol"] == 1e-10
