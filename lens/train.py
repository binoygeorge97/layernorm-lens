"""Training for the P-I surrogates (docs/plan.md, "Common rules"), float64 JAX.

- Adam (β1, β2, ε) = (0.9, 0.999, 1e-8) by default, as lens/core.py's `train`; full
  batch by default (`batch_size=None`), or minibatches of `batch_size` distinct rows
  drawn afresh at every step with jax.random.fold_in(PRNGKey(seed), step). P-I's
  default is minibatches of 2,048 (experiments/p1_quadrotor/config.yaml); the library
  default stays full batch for small synthetic problems.
- Steps run in jit-compiled chunks of `eval_every` (lax.scan); after each chunk the
  held-out one-step MSE is evaluated.
- Early stopping (plan.md): patience = ceil(patience_frac · max_steps) steps (10%),
  tolerance 1%: an evaluation counts as an improvement only if val < (1 − tol) · ref,
  where ref is the held-out MSE at the last improvement (initially at step 0).
  Training stops once `patience` steps pass without an improvement. The returned
  parameters are those with the lowest held-out MSE of any evaluation.
  `patience_frac=None` disables early stopping (a fixed budget).
- Lens logging every `log_every` steps (a multiple of `eval_every`), through
  lens/geometry.py on the first layer: see `lens_record`.
- `on_eval(step, params)`, if given, is called after every evaluation (step 0 included),
  e.g. to save parameter snapshots.
"""

import math
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from lens import analysis as an  # noqa: E402
from lens import geometry as geo  # noqa: E402
from lens import models  # noqa: E402

DEFAULTS = dict(lr=3e-3, adam=(0.9, 0.999, 1e-8), max_steps=100_000, eval_every=500,
                patience_frac=0.1, tol=0.01, batch_size=None, seed=0, log_every=500)


def lens_record(spec, p, directions=None, D=None, Z=None):
    """The first layer's lens (theory.md): degenerate flag, ‖c⊥‖, κ, z*, ‖z*‖, principal
    widths (ε-limited if degenerate, convention 4) and, for each named unit direction d
    (layer-input coordinates): r*(d), r_eff(d) along d through z*, D(d)/r_eff(d) and
    r_eff(d)/D(d) if a data half-width D(d) is given, and the susceptibility
    S(d) = ‖J(z*) d‖, the slope of the surrogate output at z* along d (D6 (g):
    ‖g′(0)‖/r*(d) = ‖J(0)‖).

    If the data inputs Z are given, the same quantities are recorded along the current
    lens's narrowest principal direction u_min (analysis.u_min), with
    D(u_min) = analysis.half_width(Z, u_min), and u_min itself."""
    E, b = models.first_layer(spec, p)
    L = geo.lens(np.asarray(E), np.asarray(b), spec.eps)
    widths = L.principal_widths_eff if L.degenerate else L.principal_widths
    rec = dict(degenerate=bool(L.degenerate), norm_c_perp=float(L.norm_c_perp), kappa=float(L.kappa),
               z_star=np.asarray(L.z_star), norm_z_star=float(np.linalg.norm(L.z_star)),
               widths=np.asarray(widths), width_min=float(widths.min()), width_max=float(widths.max()))
    zs = jnp.asarray(L.z_star)
    dirs, Ds = dict(directions or {}), dict(D or {})
    if Z is not None:
        u = an.u_min(L)
        dirs["u_min"] = u
        Ds["u_min"] = an.half_width(Z, u)[0]
        rec["u_min"] = np.asarray(u)
    for name, d in dirs.items():
        d = np.asarray(d, np.float64)
        ln = geo.line(L, d)
        S = float(jnp.linalg.norm(jax.jvp(lambda z: models.forward(spec, p, z), (zs,), (jnp.asarray(d),))[1]))
        rec[f"r_star_{name}"] = float(ln["r_star"])
        rec[f"r_eff_{name}"] = float(ln["r_eff"])
        rec[f"S_{name}"] = S
        if name in Ds:
            rec[f"D_{name}"] = float(Ds[name])
            rec[f"D_over_r_eff_{name}"] = float(Ds[name] / ln["r_eff"])
            rec[f"r_eff_over_D_{name}"] = float(ln["r_eff"] / Ds[name])
    return rec


def train(spec, p, data, cfg=None, directions=None, D=None, verbose=False, lens_Z=None, on_eval=None):
    """Train `p` on data = dict(Z, Y, Zv, Yv) (float64; Z (n, k), Y (n, n_out)).

    `lens_Z` (the training inputs, normally) adds u_min to the lens log (`lens_record`).
    Returns (best_params, history, info). history: one dict per evaluation (step,
    train_mse, val_mse, and the lens record at logging steps). info: best_step, best_val,
    stopped_step, stopped_early, patience (None if disabled), seconds, n_params, cfg."""
    c = dict(DEFAULTS, **(cfg or {}))
    if c["log_every"] % c["eval_every"]:
        raise ValueError("log_every must be a multiple of eval_every")
    f = models.batched(spec)
    Z, Y = jnp.asarray(data["Z"], jnp.float64), jnp.asarray(data["Y"], jnp.float64)
    Zv, Yv = jnp.asarray(data["Zv"], jnp.float64), jnp.asarray(data["Yv"], jnp.float64)
    n = Z.shape[0]
    b1, b2, aeps = c["adam"]
    lr, bs, chunk = c["lr"], c["batch_size"], int(c["eval_every"])
    if bs is not None and not 0 < int(bs) <= n:
        raise ValueError(f"batch_size {bs} must be in [1, n = {n}]; use None for full batch")
    bs = None if bs is None else int(bs)
    key = jax.random.PRNGKey(int(c["seed"]))

    def mse(pp, ZZ, YY):
        return jnp.mean((f(pp, ZZ) - YY) ** 2)

    @jax.jit
    def run_chunk(pp, m, v, t0):
        def step(carry, i):
            pp, m, v = carry
            t = t0 + i + 1.0
            if bs is None:
                g = jax.grad(mse)(pp, Z, Y)
            else:
                idx = jax.random.choice(jax.random.fold_in(key, t.astype(jnp.int32)), n, (bs,), replace=False)
                g = jax.grad(mse)(pp, Z[idx], Y[idx])
            m = jax.tree.map(lambda a, gg: b1 * a + (1 - b1) * gg, m, g)
            v = jax.tree.map(lambda a, gg: b2 * a + (1 - b2) * gg * gg, v, g)
            pp = jax.tree.map(lambda a, mm, vv: a - lr * (mm / (1 - b1 ** t)) / (jnp.sqrt(vv / (1 - b2 ** t)) + aeps),
                              pp, m, v)
            return (pp, m, v), None
        (pp, m, v), _ = jax.lax.scan(step, (pp, m, v), jnp.arange(chunk, dtype=jnp.float64))
        return pp, m, v

    eval_mse = jax.jit(mse)
    m = jax.tree.map(jnp.zeros_like, p)
    v = jax.tree.map(jnp.zeros_like, p)
    patience = None if c["patience_frac"] is None else int(math.ceil(c["patience_frac"] * c["max_steps"]))
    hist = [dict(step=0, train_mse=float(eval_mse(p, Z, Y)), val_mse=float(eval_mse(p, Zv, Yv)))]
    if c["log_every"]:
        hist[0].update(lens_record(spec, p, directions, D, lens_Z))
    if on_eval is not None:
        on_eval(0, p)
    best, best_step, best_p = hist[0]["val_mse"], 0, p
    ref, last_imp = best, 0
    t_start, step, stopped_early = time.time(), 0, False
    while step < c["max_steps"]:
        p, m, v = run_chunk(p, m, v, float(step))
        step += chunk
        rec = dict(step=step, train_mse=float(eval_mse(p, Z, Y)), val_mse=float(eval_mse(p, Zv, Yv)))
        if c["log_every"] and step % c["log_every"] == 0:
            rec.update(lens_record(spec, p, directions, D, lens_Z))
        hist.append(rec)
        if on_eval is not None:
            on_eval(step, p)
        if rec["val_mse"] < best:
            best, best_step, best_p = rec["val_mse"], step, p
        if rec["val_mse"] < (1.0 - c["tol"]) * ref:
            ref, last_imp = rec["val_mse"], step
        if verbose:
            print(f"  step {step}: train {rec['train_mse']:.4e} val {rec['val_mse']:.4e} best {best:.4e}@{best_step}")
        if patience is not None and step - last_imp >= patience:
            stopped_early = step < c["max_steps"]
            break
    info = dict(best_step=best_step, best_val=best, stopped_step=step, stopped_early=stopped_early,
                patience=patience, seconds=time.time() - t_start, n_params=models.n_params(p),
                cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()})
    return best_p, hist, info
