"""Analysis of trained P-I surrogates (docs/plan.md, P-I "Measure"), float64.

All coordinates are the surrogate's input coordinates (standardised (x, u) for P-I),
which is what the first layer receives. Lens quantities come from lens/geometry.py
(docs/theory.md).

- `jacobians`, `jacobian_error`: the surrogate's input-output Jacobian at each state,
  and its per-state error against a reference Jacobian.
- `lens_distances`, `near_far`: Jacobian error against lens distance: the 10% of states
  nearest the lens centre against the 50% farthest, by ρ (theory.md convention 2), or
  by ρ_eff = ‖A(z − z*)‖²/(‖c⊥‖² + Hε) when the lens is degenerate (r6.md's definition).
- `direction_sharpness`: coverage and sharpness from the weights along a direction.
- `corollary1_line`: theory.md Corollary 1 along an arbitrary line, with the line's own
  closest approach, c⊥,ℓ, κ_ℓ and r*_ℓ.
- `attenuation`: how the first layer's Lorentzian profile survives through the stack.
- `attenuation_S`: prediction 4's ratio S_out/S_block1, the output's slope at block 1's
  lens centre against that of the model truncated after block 1.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from lens import geometry as geo  # noqa: E402
from lens import models  # noqa: E402


# --------------------------------------------------------------------------- #
# Jacobian error against lens distance                                         #
# --------------------------------------------------------------------------- #


def jacobians(spec, p, Z):
    """∂y/∂z of the surrogate at each row of Z: (n, n_out, k), by forward-mode autodiff."""
    J = jax.vmap(jax.jacfwd(lambda z: models.forward(spec, p, z)))
    return np.asarray(J(jnp.asarray(Z, jnp.float64)))


def jacobian_error(J_s, J_true, kind="rel_fro"):
    """Per-state error: "rel_fro" ‖J_s − J‖_F / ‖J‖_F, or "fro" ‖J_s − J‖_F."""
    diff = np.linalg.norm((np.asarray(J_s) - np.asarray(J_true)).reshape(len(J_s), -1), axis=1)
    if kind == "fro":
        return diff
    if kind == "rel_fro":
        return diff / np.linalg.norm(np.asarray(J_true).reshape(len(J_true), -1), axis=1)
    raise ValueError(kind)


def lens_distances(L, Z):
    """(distance, kind): ρ for a non-degenerate lens, ρ_eff for a degenerate one."""
    dz = np.asarray(Z, np.float64) - L.z_star
    Adz2 = np.sum((dz @ L.A.T) ** 2, axis=-1)
    if L.degenerate:
        return Adz2 / (L.norm_c_perp ** 2 + L.H * L.eps), "rho_eff"
    return Adz2 / L.norm_c_perp ** 2, "rho"


def near_far(err, dist, near_frac=0.1, far_frac=0.5, stat="median"):
    """Error of the near_frac of states nearest the lens centre against the far_frac
    farthest (by lens distance). Sets are ceil(frac · n) states; ties broken by the
    stable sort's order. Returns the statistic of each set and their ratio near/far."""
    err, dist = np.asarray(err, np.float64), np.asarray(dist, np.float64)
    n = len(err)
    order = np.argsort(dist, kind="stable")
    n_near, n_far = int(np.ceil(near_frac * n)), int(np.ceil(far_frac * n))
    near, far = err[order[:n_near]], err[order[n - n_far:]]
    fn = {"median": np.median, "mean": np.mean}[stat]
    s_near, s_far = float(fn(near)), float(fn(far))
    return dict(n=n, n_near=n_near, n_far=n_far, stat=stat, near=s_near, far=s_far,
                ratio=s_near / s_far if s_far > 0 else np.inf,
                near_dist_max=float(dist[order[n_near - 1]]), far_dist_min=float(dist[order[n - n_far]]))


# --------------------------------------------------------------------------- #
# coverage and sharpness from the weights                                       #
# --------------------------------------------------------------------------- #


def half_width(Z, d, q=(2.5, 97.5)):
    """D(d) = (q97.5 − q2.5)/2 of (z − μ)·d, numpy's default (linear) method (as r6.md
    defines D along d₁), and whether the lens centre lies in that range is left to the
    caller."""
    Z = np.asarray(Z, np.float64)
    p = (Z - Z.mean(0)) @ np.asarray(d, np.float64)
    lo, hi = np.percentile(p, q)
    return float((hi - lo) / 2.0), float(lo), float(hi)


def direction_sharpness(L, Z, d, q=(2.5, 97.5)):
    """Along unit d through z*: D(d), r*(d), r_eff(d), sharpness D/r_eff, coverage
    (2/π)·arctan(D/r*) (theory.md, Diagnostics; with r_eff for a degenerate lens, whose
    r* is 0), and whether (z* − μ)·d lies within the data's q-range."""
    Z = np.asarray(Z, np.float64)
    d = np.asarray(d, np.float64)
    D, lo, hi = half_width(Z, d, q)
    ln = geo.line(L, d)
    width = ln["r_eff"] if L.degenerate else ln["r_star"]
    pz = float((L.z_star - Z.mean(0)) @ d)
    return dict(D=D, r_star=float(ln["r_star"]), r_eff=float(ln["r_eff"]), sharpness=D / ln["r_eff"],
                coverage=float(2.0 / np.pi * np.arctan(D / width)), z_proj=pz, z_inside=bool(lo <= pz <= hi))


def data_d1(Z):
    """The unit eigenvector of the data covariance (numpy.cov, ddof = 1) with the
    largest eigenvalue, signed so its largest-magnitude component is positive, and the
    ratio of the two largest eigenvalues (near 1 means d₁ is poorly determined)."""
    C = np.cov(np.asarray(Z, np.float64), rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    d = V[:, -1]
    d = d if d[np.argmax(np.abs(d))] >= 0 else -d
    return d, float(w[-1] / w[-2])


def u_min(L):
    """The lens's narrowest principal direction (right singular vector of A for the
    largest singular value), signed so its largest-magnitude component is positive."""
    d = L.principal_dirs[:, int(np.argmax(L.sing))]
    return d if d[np.argmax(np.abs(d))] >= 0 else -d


# --------------------------------------------------------------------------- #
# Corollary 1 on trained models                                                  #
# --------------------------------------------------------------------------- #


def corollary1_line(L, E, b, x0, d, X, n_grid=2001, t_range=3.0):
    """theory.md Corollary 1 along x0 + s·d: θ(s) = arctan((s − s*)/r*_ℓ),
    ĥ(s) = √H (cos θ ĉ_ℓ + sin θ q̂) / √(1 + κ_ℓ cos² θ), with the line's own closest
    approach s*, c⊥,ℓ, κ_ℓ and r*_ℓ. Grid: n_grid points uniform in θ over
    |θ| ≤ arctan(T/r*_ℓ), T = max(max_i |⟨x_i − x_c, d⟩|, t_range·r_eff,ℓ), x_c the
    closest-approach point (the rule of R6's D6 (g) and D7 (4)). Returns the maximum of
    ‖ĥ − ĥ_Cor1‖ / ‖ĥ_Cor1‖ over the grid with the line's quantities."""
    d, x0 = np.asarray(d, np.float64), np.asarray(x0, np.float64)
    ln = geo.line(L, d, x0)
    if ln["norm_c_perp_l"] == 0.0:
        raise ValueError("the line meets the degenerate set c_perp_l = 0.")
    xc = x0 + ln["s_star"] * d
    rs = ln["r_star"]
    T = max(float(np.max(np.abs((np.asarray(X, np.float64) - xc) @ d))), t_range * ln["r_eff"])
    th = np.linspace(-np.arctan(T / rs), np.arctan(T / rs), n_grid)
    q = L.A @ d
    qh, ch = q / np.linalg.norm(q), ln["c_perp_l"] / ln["norm_c_perp_l"]
    c = np.cos(th)[:, None]
    cf = np.sqrt(L.H) * (c * ch + np.sin(th)[:, None] * qh) / np.sqrt(1.0 + ln["kappa_l"] * c ** 2)
    pts = xc + (rs * np.tan(th))[:, None] * d
    h = pts @ np.asarray(E, np.float64).T + np.asarray(b, np.float64)
    r = h - h.mean(-1, keepdims=True)
    hh = r / np.sqrt((r ** 2).mean(-1, keepdims=True) + L.eps)
    dev = float(np.max(np.linalg.norm(hh - cf, axis=-1) / np.linalg.norm(cf, axis=-1)))
    return dict(s_star=float(ln["s_star"]), r_star_l=float(rs), r_eff_l=float(ln["r_eff"]),
                kappa_l=float(ln["kappa_l"]), T=float(T), n_grid=int(n_grid), max_rel_dev=dev)


# --------------------------------------------------------------------------- #
# attenuation in the stack                                                       #
# --------------------------------------------------------------------------- #


def attenuation(spec, p, L, d, X, n_grid=4001, t_range=3.0):
    """Along z* + t·d, on a grid uniform in t over |t| ≤ T, T = max(data extent along d
    from z*, t_range·r_eff(d)): the speed profiles ‖∂ĥ_j/∂t‖ of each block's pre-affine
    LayerNorm output and ‖∂y/∂t‖ of the output, each summarised by its sharpness
    peak/median over the grid. attenuation_j = sharpness_j / sharpness_1 for blocks
    j ≥ 1 and for the output (block 1's profile is the Lorentzian of Corollary 1)."""
    d = jnp.asarray(d, jnp.float64)
    zs = jnp.asarray(L.z_star, jnp.float64)
    ln = geo.line(L, np.asarray(d))
    T = max(float(np.max(np.abs((np.asarray(X, np.float64) - L.z_star) @ np.asarray(d)))), t_range * ln["r_eff"])
    t = np.linspace(-T, T, n_grid)

    def speeds(tt):
        y_dot, hats_dot = jax.jvp(lambda s: models.trace(spec, p, zs + s * d), (tt,), (1.0,))[1]
        return jnp.linalg.norm(y_dot), jnp.stack([jnp.linalg.norm(h) for h in hats_dot])

    out, blocks = jax.vmap(speeds)(jnp.asarray(t))
    out, blocks = np.asarray(out), np.asarray(blocks)          # (n,), (n, n_blocks)
    sharp = lambda v: float(np.max(v) / np.median(v))
    s_blocks = [sharp(blocks[:, j]) for j in range(blocks.shape[1])]
    s_out = sharp(out)
    return dict(T=float(T), n_grid=int(n_grid), sharpness_blocks=s_blocks, sharpness_out=s_out,
                attenuation_blocks=[s / s_blocks[0] for s in s_blocks], attenuation_out=s_out / s_blocks[0])


def attenuation_S(spec, p, L, d):
    """Prediction 4 (p1 draft, D10): along the line z* + t·d through block 1's lens
    centre, S_out = ‖∂y/∂t‖ at t = 0 for the full model and S_block1 = the same for the
    model truncated after block 1 (models.trace(..., upto=1): the head applied to block
    1's output). The ratio S_out/S_block1 is below 1 when the later blocks attenuate the
    slope the first block's lens puts at its centre. S is the peak slope at the lens
    centre (theory.md, Diagnostics: the susceptibility)."""
    zs, dd = jnp.asarray(L.z_star, jnp.float64), jnp.asarray(d, jnp.float64)
    slope = lambda fn: float(jnp.linalg.norm(jax.jvp(fn, (zs,), (dd,))[1]))  # noqa: E731
    s_out = slope(lambda z: models.trace(spec, p, z)[0])
    s_1 = slope(lambda z: models.trace(spec, p, z, upto=1)[0])
    return dict(S_out=s_out, S_block1=s_1, ratio=s_out / s_1 if s_1 > 0 else np.inf)
