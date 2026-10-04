"""lens/models.py and lens/train.py on synthetic data only (no quadrotor data)."""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lens import geometry as geo  # noqa: E402
from lens import models  # noqa: E402
from lens import train as tr  # noqa: E402


def _spec(**kw):
    base = dict(arch="prenorm", n_blocks=1, H=8, k=3, n_out=2, eps=1e-5, init="torch_default", branch_width=6)
    base.update(kw)
    return models.SurrogateSpec(**base)


@pytest.mark.parametrize("arch", models.ARCHS)
@pytest.mark.parametrize("n_blocks", [1, 3])
def test_shapes_and_first_layer(arch, n_blocks):
    spec = _spec(arch=arch, n_blocks=n_blocks)
    p = models.init_params(spec, 0)
    E, b = models.first_layer(spec, p)
    assert E.shape == (8, 3) and b.shape == (8,)
    y, hats = models.trace(spec, p, jnp.ones(3))
    assert y.shape == (2,) and len(hats) == n_blocks and all(h.shape == (8,) for h in hats)
    assert models.batched(spec)(p, jnp.ones((5, 3))).shape == (5, 2)
    for h in hats:  # pre-affine LayerNorm output: mean 0, mean square < 1 (eps)
        assert abs(float(jnp.mean(h))) < 1e-12 and float(jnp.mean(h ** 2)) < 1.0


@pytest.mark.parametrize("arch", models.ARCHS)
def test_initialisations(arch):
    spec = _spec(arch=arch, n_blocks=3, H=64, k=16, n_out=12, branch_width=64)
    pz = models.init_params(spec.__class__(**{**spec.__dict__, "init": "zero_bias"}), 1)
    pt = models.init_params(spec, 1)
    for name in pt:
        if name.startswith(("g",)) and pt[name].ndim == 1:
            assert np.all(np.asarray(pt[name]) == 1.0)
    for name, W in pt.items():
        if W.ndim == 2:  # weights U(-1/sqrt(fan_in), 1/sqrt(fan_in)) in both inits
            lim = 1 / np.sqrt(W.shape[1])
            assert np.max(np.abs(W)) <= lim and np.std(np.asarray(W)) == pytest.approx(lim / np.sqrt(3), rel=0.15)
            assert np.array_equal(np.asarray(W), np.asarray(pz[name]))  # same weights in both inits
    biases = [n for n, W in pt.items() if W.ndim == 1 and not n.startswith(("g", "be"))]
    assert biases and all(np.all(np.asarray(pz[n]) == 0) for n in biases)
    assert all(np.any(np.asarray(pt[n]) != 0) for n in biases)
    E, b = models.first_layer(spec, pz)  # zero bias: degenerate lens at the origin (Remark 1)
    L = geo.lens(np.asarray(E), np.asarray(b), spec.eps)
    assert L.degenerate and np.allclose(L.z_star, 0)


def test_forward_matches_a_hand_written_pass():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(3)

    def ln(h, eps=1e-5):
        r = h - h.mean()
        return r / np.sqrt((r ** 2).mean() + eps)

    from scipy.special import erf
    gelu = lambda x: 0.5 * x * (1 + erf(x / np.sqrt(2)))
    mish = lambda x: x * np.tanh(np.log1p(np.exp(x)))
    spec = _spec(arch="prenorm", n_blocks=2)
    p = {k: np.asarray(v) for k, v in models.init_params(spec, 3).items()}
    h = p["E"] @ z + p["b"]
    for j in range(2):
        h = h + p[f"W2_{j}"] @ gelu(p[f"W1_{j}"] @ (p[f"g{j}"] * ln(h) + p[f"be{j}"]) + p[f"b1_{j}"]) + p[f"b2_{j}"]
    want = p["Wo"] @ h + p["bo"]
    assert np.allclose(models.forward(spec, models.init_params(spec, 3), jnp.asarray(z)), want, atol=1e-13)
    spec = _spec(arch="normedlinear", n_blocks=2)
    p = {k: np.asarray(v) for k, v in models.init_params(spec, 4).items()}
    x = z
    for j in range(2):
        x = mish(p[f"g{j}"] * ln(p[f"W{j}"] @ x + p[f"b{j}"]) + p[f"be{j}"])
    want = p["Wo"] @ x + p["bo"]
    assert np.allclose(models.forward(spec, models.init_params(spec, 4), jnp.asarray(z)), want, atol=1e-13)


def _linear_data(n=400, k=3, n_out=2, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((k, n_out))
    Z, Zv = rng.standard_normal((n, k)), rng.standard_normal((n // 4, k))
    return dict(Z=Z, Y=np.tanh(Z) @ W, Zv=Zv, Yv=np.tanh(Zv) @ W)


@pytest.mark.parametrize("batch_size", [None, 64])
def test_training_fits_a_synthetic_target(batch_size):
    spec = _spec(arch="normedlinear", n_blocks=1, H=32)
    data = _linear_data()
    p0 = models.init_params(spec, 0)
    p, hist, info = tr.train(spec, p0, data, dict(lr=1e-2, max_steps=2000, eval_every=200, log_every=0,
                                                 batch_size=batch_size))
    assert hist[-1]["val_mse"] < 0.1 * hist[0]["val_mse"]
    # the returned parameters are the best evaluation's
    f = models.batched(spec)
    assert info["best_val"] == min(h["val_mse"] for h in hist)
    assert float(jnp.mean((f(p, data["Zv"]) - data["Yv"]) ** 2)) == pytest.approx(info["best_val"], rel=1e-12)


def test_early_stopping_patience_and_tolerance():
    spec = _spec()
    data = _linear_data()
    p0 = models.init_params(spec, 0)
    # lr = 0: no evaluation ever improves; patience = 10% of 10,000 = 1,000 steps
    p, hist, info = tr.train(spec, p0, data, dict(lr=0.0, max_steps=10_000, eval_every=250, log_every=0))
    assert info["patience"] == 1000 and info["stopped_step"] == 1000 and info["stopped_early"]
    assert info["best_step"] == 0 and all(np.array_equal(p[k], p0[k]) for k in p)
    # without early stopping triggered, training runs to max_steps
    p, hist, info = tr.train(spec, p0, data, dict(lr=1e-2, max_steps=1000, eval_every=250, log_every=0,
                                                 patience_frac=10.0))
    assert info["stopped_step"] == 1000 and not info["stopped_early"]


def test_lens_logging():
    spec = _spec(arch="prenorm", n_blocks=2, init="zero_bias")
    data = _linear_data()
    d = np.array([1.0, 0.0, 0.0])
    p0 = models.init_params(spec, 0)
    p, hist, info = tr.train(spec, p0, data, dict(lr=1e-2, max_steps=600, eval_every=200, log_every=200),
                             directions=dict(d1=d), D=dict(d1=2.0))
    logged = [h for h in hist if "z_star" in h]
    assert [h["step"] for h in logged] == [0, 200, 400, 600]
    assert logged[0]["degenerate"] and logged[0]["norm_z_star"] == 0.0  # zero-bias start (Remark 1)
    assert not logged[-1]["degenerate"]  # training moves the biases off zero
    rec = tr.lens_record(spec, p0, dict(d1=d), dict(d1=2.0))
    E, b = models.first_layer(spec, p0)
    L = geo.lens(np.asarray(E), np.asarray(b), spec.eps)
    assert rec["r_eff_d1"] == pytest.approx(geo.line(L, d)["r_eff"]) and rec["r_star_d1"] == 0.0
    assert rec["D_over_r_eff_d1"] == pytest.approx(2.0 / rec["r_eff_d1"])
    S = np.linalg.norm(np.asarray(jax.jvp(lambda z: models.forward(spec, p0, z), (jnp.asarray(L.z_star),),
                                          (jnp.asarray(d),))[1]))
    assert rec["S_d1"] == pytest.approx(S)


def test_minibatch_size_is_checked_and_patience_can_be_disabled():
    spec = _spec()
    data = _linear_data()
    p0 = models.init_params(spec, 0)
    with pytest.raises(ValueError):
        tr.train(spec, p0, data, dict(max_steps=100, eval_every=50, log_every=0, batch_size=401))
    # patience_frac None: a fixed budget, even with lr = 0 (no evaluation ever improves)
    seen = []
    p, hist, info = tr.train(spec, p0, data, dict(lr=0.0, max_steps=1000, eval_every=250, log_every=0,
                                                 patience_frac=None, batch_size=64),
                             on_eval=lambda step, pp: seen.append(step))
    assert info["patience"] is None and info["stopped_step"] == 1000 and not info["stopped_early"]
    assert seen == [0, 250, 500, 750, 1000]


def test_minibatches_are_the_same_for_both_initialisations_of_a_seed():
    """The minibatch stream depends only on the run's seed and the step, so the paired
    arms (shared weights) also see the same batches."""
    data = _linear_data()
    out = []
    for init in ("zero_bias", "torch_default"):
        spec = _spec(init=init)
        p0 = models.init_params(spec, 2)
        p0 = dict(p0, b=jnp.zeros_like(p0["b"]), b1_0=jnp.zeros_like(p0["b1_0"]), b2_0=jnp.zeros_like(p0["b2_0"]),
                  bo=jnp.zeros_like(p0["bo"]))
        p, _, _ = tr.train(spec, p0, data, dict(lr=1e-2, max_steps=200, eval_every=100, log_every=0, batch_size=32,
                                                patience_frac=None, seed=2))
        out.append(p)
    assert all(np.array_equal(out[0][k], out[1][k]) for k in out[0])


def test_lens_record_along_u_min():
    from lens import analysis as an
    spec = _spec(H=16, k=3)
    p0 = models.init_params(spec, 1)
    Z = np.random.default_rng(0).uniform(-np.sqrt(3), np.sqrt(3), (500, 3))
    rec = tr.lens_record(spec, p0, None, None, Z)
    E, b = models.first_layer(spec, p0)
    L = geo.lens(np.asarray(E), np.asarray(b), spec.eps)
    u = an.u_min(L)
    assert np.array_equal(rec["u_min"], u)
    assert rec["D_u_min"] == an.half_width(Z, u)[0]
    assert rec["r_eff_u_min"] == pytest.approx(np.sqrt(L.norm_c_perp ** 2 + L.H * L.eps) / L.sing[0], rel=1e-12)
    assert rec["r_eff_over_D_u_min"] == pytest.approx(rec["r_eff_u_min"] / rec["D_u_min"], rel=1e-15)
    assert rec["r_eff_over_D_u_min"] * rec["D_over_r_eff_u_min"] == pytest.approx(1.0, rel=1e-14)
    # u_min is the narrowest principal direction: no unit direction has a smaller r_eff
    d = np.random.default_rng(1).standard_normal((200, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    assert min(geo.line(L, dj)["r_eff"] for dj in d) >= rec["r_eff_u_min"] * (1 - 1e-12)
    # in the training history, every logged step carries u_min
    data = _linear_data()
    _, hist, _ = tr.train(spec, p0, data, dict(lr=1e-2, max_steps=200, eval_every=100, log_every=100), lens_Z=data["Z"])
    assert all("r_eff_over_D_u_min" in h and h["u_min"].shape == (3,) for h in hist)
