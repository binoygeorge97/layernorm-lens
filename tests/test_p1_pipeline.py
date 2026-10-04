"""P-I analysis, LQR, hover check, data and stages, on synthetic plants only.

No surrogate is trained on quadrotor data and no lens quantity is computed on a
surrogate of the quadrotor: the end-to-end tests use a synthetic linear plant with the
quadrotor's dimensions (12 states, 4 inputs)."""

import copy
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
import data as pdata  # noqa: E402
import hover as phover  # noqa: E402
import run as prun  # noqa: E402
from control import lqr  # noqa: E402
from lens import analysis as an  # noqa: E402
from lens import geometry as geo  # noqa: E402
from lens import models  # noqa: E402

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
    cfg["training"].update(max_steps=400, eval_every=100, log_every=200)
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
    r = prun.hover_one(_small_cfg(), spec, p, ds, plant, hx, hu)
    assert r["rel_err_A"] < 1e-10 and r["rel_err_B"] < 1e-10 and r["rel_err_K"] < 1e-8
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
    for name, spec, _ in members:
        assert (tmp_path / "res" / "history" / f"{name}.csv").exists()
        h = np.load(tmp_path / "res" / "history" / f"{name}.npz")
        assert list(h["step"]) == [0, 200, 400] and h["z_star"].shape == (3, 16)
        p = prun.load_params(str(tmp_path / "ck" / f"{name}.npz"))
        out = prun.analyse_one(cfg, spec, p, ds)
        assert {"nf_ratio", "d1_sharpness", "u_min_coverage", "err_median"} <= set(out)
        if not out["degenerate"]:
            assert out["corollary1_max_rel_dev"] < 1e-10 and "attenuation_out" in out
        r = prun.hover_one(cfg, spec, p, ds, plant, hx, hu)
        assert {"rel_err_A", "rel_err_B", "sign_agree_A"} <= set(r)


def test_benchmark_stage_runs_on_random_targets(tmp_path):
    cfg = _small_cfg()
    cfg["benchmark"].update(n_train=200, n_val=50)
    members = [m for m in prun.grid(cfg) if m[2] == 0][:2]
    rows, summary = prun.stage_benchmark(cfg, str(tmp_path), members, steps=200)
    assert len(rows) == 2 and all(r["seconds_per_step"] > 0 for r in rows)
    assert (tmp_path / "benchmark.csv").exists() and summary["n_models"] == 2


@pytest.mark.parametrize("stage", prun.GATED)
def test_gated_stages_refuse_without_the_prereg_tag(stage):
    r = subprocess.run([sys.executable, os.path.join(P1, "run.py"), "--config", os.path.join(P1, "config.yaml"), stage],
                       capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode != 0 and "prereg-p1" in (r.stderr + r.stdout)
