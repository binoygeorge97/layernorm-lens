"""Fixed-seed cases for the lens/core.py regression guard.

Shared by make_golden.py (writes tests/golden/*.npz) and
tests/test_core_regression.py (recomputes and compares), so the two cannot drift.

Each case returns a flat dict {name: np.ndarray}. Long arrays are also stored as a
SHA-256 of their bytes (`<name>#sha256`), which the bit-exact mode checks, plus a
thinned copy (`<name>#thin`) that the tolerance mode compares.
"""

import hashlib

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lens import core

THIN = 53  # keep every THIN-th row of long arrays
LONG = 2000  # arrays with more elements than this are hashed and thinned

GRID = dict(n_uniform=4001, n_local=2001)


def _flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k in sorted(obj):
            _flatten(f"{prefix}{k}/", obj[k], out)
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
        for i, o in enumerate(obj):
            _flatten(f"{prefix}{i}/", o, out)
    else:
        a = np.asarray(obj)
        name = prefix[:-1]
        if a.dtype.kind in "fciub" and a.size > LONG:
            out[f"{name}#sha256"] = np.array(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
            out[f"{name}#shape"] = np.array(a.shape)
            out[f"{name}#thin"] = a[::THIN]
        else:
            out[name] = a
    return out


def flat(obj):
    return _flatten("", obj, {})


# --------------------------------------------------------------------------- #


def case_norm():
    h = jax.random.normal(jax.random.PRNGKey(0), (7, 16), dtype=jnp.float64)
    out = {}
    for kind in ("layer", "rms", "static", "taper", "none"):
        for eps in (0.0, 1e-6, 1e-2):
            out[f"{kind}/{eps:g}"] = core.norm_apply(kind, h, eps, sigma0=1.3, gate=0.4)
    return out


def case_init():
    out = {"block": {}, "stack": {}}
    for br in ("mlp", "attn", "ssm"):
        out["block"][br] = core.init_params(3, H=16, WB=32, n_in=2, branch=br)
    out["stack"]["mlp3"] = core.init_stack(4, n_blocks=3, H=16, n_in=3, branch="mlp")
    return out


def case_block_fwd():
    out = {}
    z = jax.random.normal(jax.random.PRNGKey(1), (11, 2), dtype=jnp.float64)
    for br in ("mlp", "attn", "ssm"):
        p = core.init_params(5, H=16, WB=32, n_in=2, branch=br)
        p["b"] = 0.1 * jnp.arange(16.0)
        ctx = 0.3 * jnp.ones((3, 16))
        for norm in ("layer", "rms", "static", "taper"):
            fwd, nhat, mvec = core.make_block(norm, br, H=16, eps=1e-6)
            F = jax.vmap(lambda zz: fwd(p, zz, ctx, 1.1, 0.5))(z)
            V = jax.vmap(lambda zz: nhat(p, zz, 1.1, 0.5))(z)
            M = jax.vmap(lambda v: mvec(p, v, ctx))(V)
            out[f"{br}/{norm}"] = dict(F=F, V=V, M=M)
        fwd, _, _ = core.make_block("layer", br, H=16, eps=1e-6, postnorm=True)
        out[f"{br}/postnorm"] = jax.vmap(lambda zz: fwd(p, zz, ctx))(z)
    return out


def case_stack():
    out = {}
    p = core.init_stack(6, n_blocks=3, H=16, n_in=3, branch="mlp")
    p["b"] = 0.05 * jnp.arange(16.0) - 0.4
    fwd, nhat_k, h_upto = core.make_stack(3, "layer", "mlp", H=16, eps=1e-6)
    z = jax.random.normal(jax.random.PRNGKey(2), (9, 3), dtype=jnp.float64)
    for fr in ((), (0,), (1,), (0, 2)):
        out[f"fwd/{fr}"] = jax.vmap(lambda zz: fwd(p, zz, None, fr))(z)
        out[f"J/{fr}"] = jax.vmap(jax.grad(lambda zz: fwd(p, zz, None, fr)))(z)
    for k in range(4):
        out[f"nhat/{k}"] = jax.vmap(lambda zz: nhat_k(p, zz, k))(z)
    d = jnp.array([0.6, -0.8, 0.0])
    z0 = jnp.array([0.2, 0.1, -0.3])
    ctx = jnp.zeros((0, 16))

    def h_of_s(s):
        return h_upto(p, z0 + s * d, 2, ctx, frozenset())

    for s0 in (0.0, 0.7):
        out[f"local/{s0}"] = core.local_geometry(h_of_s, s0, H=16, eps=1e-6)
    return out


def case_geometry():
    out = {}
    p = core.init_params(7, H=16, n_in=3, branch="mlp")
    p["b"] = jax.random.uniform(jax.random.PRNGKey(8), (16,), minval=-0.5, maxval=0.5,
                                dtype=jnp.float64)
    d = np.array([0.3, -0.5, 0.2])
    d /= np.linalg.norm(d)
    z0 = np.array([0.4, 0.1, -1.0])
    s = np.linspace(-3.0, 3.0, 41)
    for eps in (0.0, 1e-6, 1e-2):
        for centred in (True, False):
            g = core.geometry(p, d, z0, H=16, eps=eps, centred=centred)
            tag = f"{eps:g}/{'layer' if centred else 'rms'}"
            out[f"{tag}/g"] = g
            out[f"{tag}/dnhat"] = core.dnhat_closed(g, s, H=16, eps=eps)
            out[f"{tag}/lorentz"] = core.lorentzian_norm(g, s, H=16)
            out[f"{tag}/profile"] = core.profile_norm(g, s, H=16, eps=eps)
    out["hwhm_ratio_pred"] = np.array([core.hwhm_ratio_pred(f) for f in np.linspace(0, 1, 11)])
    ys = 1.0 / (1.0 + ((s - 0.3) / 0.7) ** 2)
    out["hwhm"] = np.array(core.hwhm(s, ys))
    out["tv"] = np.array(core.total_variation(s, ys))
    out["sweep_grid"] = core.sweep_grid(out["1e-06/layer/g"], 4.0, 101, 51)
    out["common_grid"] = core.common_grid([out["1e-06/layer/g"], out["0.01/rms/g"]], 4.0, 101, 51)
    return out


def _eval_one(norm, br, seed, bias_scale):
    cfg = dict(H=16, eps=1e-6, norm=norm, branch=br, **GRID)
    p = core.init_params(seed, H=16, n_in=2, branch=br)
    p["b"] = bias_scale * jax.random.normal(jax.random.PRNGKey(seed + 100), (16,),
                                            dtype=jnp.float64)
    d = np.array([1.0, 0.0])
    z0 = np.array([0.0, 0.5])
    out, r = core.evaluate(p, cfg, d, z0, X=4.0, sigma0=1.2, gate=1.0)
    out = {k: v for k, v in out.items()}
    return dict(out=out, r=r)


def case_evaluate():
    res = {}
    for norm in ("layer", "rms", "static"):
        res[f"{norm}/mlp"] = _eval_one(norm, "mlp", 11, 0.3)
    res["layer/attn"] = _eval_one("layer", "attn", 12, 0.3)
    res["layer/ssm"] = _eval_one("layer", "ssm", 13, 0.3)
    res["layer/mlp/zero_bias"] = _eval_one("layer", "mlp", 14, 0.0)
    cfg = dict(H=16, eps=1e-6, norm="layer", branch="mlp", postnorm=True, **GRID)
    p = core.init_params(15, H=16, n_in=2, branch="mlp")
    o, r = core.evaluate(p, cfg, np.array([1.0, 0.0]), np.array([0.0, 0.5]), X=4.0)
    res["layer/mlp/postnorm"] = dict(out=o, r=r)
    return res


def case_data():
    out = {}
    for pl in ("linear", "sinc"):
        out[pl] = core.make_data(pl, n=500, seed=3)
    dl = core.make_data("linear", n=300, seed=4, L=4)
    out["linear_L4"] = dl
    dm = core.make_data_mimo(n=400, seed=5)
    out["mimo"] = dm
    ud, th = core.unit_directions(8)
    out["unit_directions"] = dict(d=ud, theta=th)
    out["var_ratio"] = np.array(core.var_ratio(out["sinc"]))
    return {k: ({kk: vv for kk, vv in v.items() if vv is not None} if isinstance(v, dict) else v)
            for k, v in out.items()}


def case_train():
    out = {}
    data = core.make_data("sinc", n=1000, seed=21)
    cfg = dict(H=16, eps=1e-6, norm="layer", branch="mlp", steps=400, chunk=200, lr=3e-3)
    p = core.init_params(22, H=16, n_in=2, branch="mlp")
    p, hist = core.train(p, data, cfg, verbose=False)
    out["mlp"] = dict(p=p, loss=np.array([h["loss"] for h in hist]),
                      frozen_sigma=np.array(core.frozen_sigma(p, data, cfg)))

    data = core.make_data("linear", n=300, seed=23, L=3)
    cfg = dict(H=16, eps=1e-6, norm="layer", branch="attn", steps=200, chunk=100, lr=3e-3)
    p = core.init_params(24, H=16, n_in=2, branch="attn", L=3)
    p, hist = core.train(p, data, cfg, verbose=False,
                         extra_loss=lambda pp: 1e-4 * jnp.sum(pp["E"] ** 2))
    out["attn_ctx"] = dict(p=p, loss=np.array([h["loss"] for h in hist]))
    return out


CASES = {
    "norm": case_norm,
    "init": case_init,
    "block_fwd": case_block_fwd,
    "stack": case_stack,
    "geometry": case_geometry,
    "evaluate": case_evaluate,
    "data": case_data,
    "train": case_train,
}


def compute(name):
    return flat(CASES[name]())


def versions():
    import jaxlib
    import scipy
    return dict(jax=jax.__version__, jaxlib=jaxlib.__version__,
                numpy=np.__version__, scipy=scipy.__version__,
                x64=bool(jax.config.read("jax_enable_x64")),
                backend=jax.default_backend())
