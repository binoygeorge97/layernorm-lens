"""kink_core - E0 evaluator for the pre-norm kink (ICLR 2027 sprint).

One module, imported unchanged by every sprint kernel, so that E1/E2/E3 tables
are produced by the same code path and cannot drift apart overnight.

Conventions follow `envelope_violation_check` / `sweep_core`:
  block   F(x,u) = Wd.(E z + b) + <Wd, branch(nhat(E z + b))> + bd,  z = (x,u)
  sweep   x varies, u (and any context tokens) held fixed
  units   all derivatives are of the DISCRETE map; float64 throughout

Geometry (Theorem 1), for a sweep x -> x0 + s d:
  q      = P E^T d                    P = I - 11^T/H  (layer) or I (rms)
  c      = P (E^T z0 + b)
  s*     = -<q,c>/|q|^2               offset of the closest-approach point
  c_perp = c + s* q                   (orthogonal to q by construction)
  D      = sqrt(|c_perp|^2 + H eps)
  delta  = D / sqrt(H)                r* = D / |q|          peak|dnhat/ds| = |q|/delta
  peak x width = sqrt(H)              (invariance, Theorem 2)

Amplitude (Proposition 3):  J_branch(x*) = <m(x*), q> / delta,
  m(x) = grad_v [ <Wd, branch(v)> ] at v = nhat(x), i.e. ONE VJP.
  The branch enters here and nowhere else, which is the branch-agnostic corollary.

Nothing in `geometry` touches the branch parameters. That is the point.
"""

import os
import time
import json
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

L_PHI = 1.128904145  # Phi(sqrt2) + sqrt2 phi(sqrt2), the GELU' bound

# --------------------------------------------------------------------------- #
# 1. normalisations                                                            #
# --------------------------------------------------------------------------- #
# Every arm is (centring?) -> divide by a scale.  `static` freezes the scale so
# the map becomes degree-one in x and no closest-approach point exists; `taper`
# interpolates, which is E5's gate g.


def _center(h):
    return h - jnp.mean(h, -1, keepdims=True)


def norm_apply(kind, h, eps, sigma0=1.0, gate=1.0):
    """kind in {layer, rms, static, taper, none}. sigma0 is the frozen scale."""
    if kind == "none":
        return h
    if kind == "layer":
        r = _center(h)
        return r / jnp.sqrt(jnp.mean(r**2, -1, keepdims=True) + eps)
    if kind == "rms":
        return h / jnp.sqrt(jnp.mean(h**2, -1, keepdims=True) + eps)
    if kind == "static":
        return _center(h) / sigma0
    if kind == "taper":
        r = _center(h)
        live = jnp.sqrt(jnp.mean(r**2, -1, keepdims=True) + eps)
        return r / (gate * live + (1.0 - gate) * sigma0)
    raise ValueError(kind)


CENTRED = {"layer": True, "static": True, "taper": True, "rms": False, "none": True}
HAS_KINK = {"layer": True, "rms": True, "taper": True, "static": False, "none": False}


# --------------------------------------------------------------------------- #
# 2. branches.  interface: params -> (v_swept, ctx) -> R^H                      #
# --------------------------------------------------------------------------- #
# v_swept : (H,)      the normalised token whose pre-norm input is affine in x
# ctx     : (L-1, H)  normalised context tokens, constant along the sweep
# Only the map v_swept -> R^H matters for m(x); ctx enters as a constant.


def _glorot(key, shape):
    lim = np.sqrt(6.0 / (shape[0] + shape[1]))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim, dtype=jnp.float64)


def branch_init(kind, key, H, WB, heads=4, L=4):
    k = jax.random.split(key, 8)
    if kind == "mlp":
        return dict(W1=_glorot(k[0], (H, WB)), b1=jnp.zeros(WB),
                    W2=_glorot(k[1], (WB, H)), b2=jnp.zeros(H))
    if kind == "attn":
        dh = H // heads
        assert dh * heads == H, "H must divide by heads"
        return dict(Wq=_glorot(k[0], (H, H)), Wk=_glorot(k[1], (H, H)),
                    Wv=_glorot(k[2], (H, H)), Wo=_glorot(k[3], (H, H)),
                    b_o=jnp.zeros(H), pos=0.02 * jax.random.normal(k[4], (L, H), dtype=jnp.float64))
    if kind == "ssm":
        return dict(Win=_glorot(k[0], (H, WB)), a=jnp.zeros(WB) + 0.5,
                    B=jnp.ones(WB), Wout=_glorot(k[1], (WB, H)), b_o=jnp.zeros(H))
    raise ValueError(kind)


def branch_apply(kind, bp, v, ctx, heads=4):
    """Returns R^H. `v` is the swept token, `ctx` the (L-1,H) frozen prefix."""
    if kind == "mlp":
        a = jax.nn.gelu(v @ bp["W1"] + bp["b1"], approximate=False)
        return a @ bp["W2"] + bp["b2"]
    if kind == "attn":
        toks = jnp.concatenate([ctx, v[None, :]], 0)  # (L,H), swept token last
        toks = toks + bp["pos"][: toks.shape[0]]
        L, H = toks.shape
        dh = H // heads
        q = (toks @ bp["Wq"]).reshape(L, heads, dh)
        k_ = (toks @ bp["Wk"]).reshape(L, heads, dh)
        v_ = (toks @ bp["Wv"]).reshape(L, heads, dh)
        # causal read-out at the last position only
        logits = jnp.einsum("hd,lhd->hl", q[-1], k_) / jnp.sqrt(dh)
        w = jax.nn.softmax(logits, axis=-1)
        o = jnp.einsum("hl,lhd->hd", w, v_).reshape(H)
        return o @ bp["Wo"] + bp["b_o"]
    if kind == "ssm":
        toks = jnp.concatenate([ctx, v[None, :]], 0)  # (L,H)
        A = jnp.exp(-jax.nn.softplus(bp["a"]))  # (WB,), in (0,1)
        inp = toks @ bp["Win"] * bp["B"]  # (L,WB)

        def step(s, u):
            s = A * s + u
            return s, s

        s_last, _ = jax.lax.scan(step, jnp.zeros_like(A), inp)
        return jax.nn.gelu(s_last, approximate=False) @ bp["Wout"] + bp["b_o"]
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# 3. the block                                                                  #
# --------------------------------------------------------------------------- #


def init_params(seed, H=16, WB=32, n_in=2, branch="mlp", heads=4, L=4):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    p = dict(E=_glorot(k[0], (n_in, H)), b=jnp.zeros(H),
             gamma=jnp.ones(H), beta=jnp.zeros(H),
             Wd=_glorot(k[1], (H, 1)), bd=jnp.zeros(1))
    p["br"] = branch_init(branch, k[2], H, WB, heads=heads, L=L)
    return p


def enc(p, z):
    """z: (..., n_in) -> pre-norm activation (..., H). Affine: assumption A1."""
    return z @ p["E"] + p["b"]


def make_block(kind_norm, kind_branch, H=16, eps=1e-6, heads=4, postnorm=False):
    """Returns (fwd, nhat, mvec) closures. All take (p, z, ctx_z, sigma0, gate)."""

    def nhat(p, z, sigma0=1.0, gate=1.0):
        return norm_apply(kind_norm, enc(p, z), eps, sigma0, gate)

    def branch_scalar(p, v, ctx):
        return jnp.dot(branch_apply(kind_branch, p["br"], p["gamma"] * v + p["beta"],
                                    ctx, heads=heads), p["Wd"][:, 0])

    def fwd(p, z, ctx=None, sigma0=1.0, gate=1.0):
        h = enc(p, z)
        if ctx is None:
            ctx = jnp.zeros((0, H), dtype=jnp.float64)
        if postnorm:
            # textbook post-norm: F = <Wd, gamma*N(h + branch(h)) + beta> + bd.
            # There is no identity path outside the norm, which is Lemma 2's
            # global collapse; a_skip is NOT defined for this arm.
            bv = branch_apply(kind_branch, p["br"], h, ctx, heads=heads)
            v = norm_apply("layer", h + bv, eps)
            return jnp.dot(p["gamma"] * v + p["beta"], p["Wd"][:, 0]) + p["bd"][0]
        v = norm_apply(kind_norm, h, eps, sigma0, gate)
        return jnp.dot(h, p["Wd"][:, 0]) + branch_scalar(p, v, ctx) + p["bd"][0]

    def mvec(p, v, ctx):
        """m(x) = grad_v <Wd, branch(v)>. ONE VJP; the only place the branch enters."""
        return jax.grad(branch_scalar, argnums=1)(p, v, ctx)

    return fwd, nhat, mvec


# --------------------------------------------------------------------------- #
# 3b. stacked blocks - Remark 5 / assumption A1 beyond the first block           #
# --------------------------------------------------------------------------- #
# A1 (affine pre-norm input) is EXACT only for block 1. For block k>1 the norm
# input is a curved function of the embedding, so Theorem 1 holds locally. The
# `freeze` argument stop-gradients a block's branch input, which removes that
# block's contribution to dF/dx exactly, giving a clean ablation:
#   freeze=(1,)  -> block-1 kink only
#   freeze=(0,)  -> block-2 kink only, and h1 is then affine in x, so A1 is
#                   exact for block 2 and its geometry is closed-form too.


def init_stack(seed, n_blocks=2, H=16, WB=32, n_in=2, branch="mlp", heads=4, L=4):
    k = jax.random.split(jax.random.PRNGKey(seed), 3 + n_blocks)
    p = dict(E=_glorot(k[0], (n_in, H)), b=jnp.zeros(H),
             Wd=_glorot(k[1], (H, 1)), bd=jnp.zeros(1))
    for j in range(n_blocks):
        p[f"gamma{j}"] = jnp.ones(H)
        p[f"beta{j}"] = jnp.zeros(H)
        p[f"br{j}"] = branch_init(branch, k[3 + j], H, WB, heads=heads, L=L)
    return p


def make_stack(n_blocks, kind_norm, kind_branch, H=16, eps=1e-6, heads=4):
    def h_upto(p, z, j, ctx, freeze):
        h = enc(p, z)
        for i in range(j):
            v = norm_apply(kind_norm, h, eps)
            if i in freeze:
                v = jax.lax.stop_gradient(v)
            bv = branch_apply(kind_branch, p[f"br{i}"],
                              p[f"gamma{i}"] * v + p[f"beta{i}"], ctx, heads=heads)
            h = h + bv
        return h

    def fwd(p, z, ctx=None, freeze=()):
        if ctx is None:
            ctx = jnp.zeros((0, H), dtype=jnp.float64)
        h = h_upto(p, z, n_blocks, ctx, frozenset(freeze))
        return jnp.dot(h, p["Wd"][:, 0]) + p["bd"][0]

    def nhat_k(p, z, k, ctx=None, freeze=()):
        if ctx is None:
            ctx = jnp.zeros((0, H), dtype=jnp.float64)
        return norm_apply(kind_norm, h_upto(p, z, k, ctx, frozenset(freeze)), eps)

    return fwd, nhat_k, h_upto


def local_geometry(h_of_s, s0, H=16, eps=1e-6, centred=True):
    """Local-affine geometry of a CURVED pre-norm input h(s) at the point s0.

    Linearises h at s0 and applies Theorem 1 to the tangent line. Returns the
    same fields as `geometry` plus `curv`, the second derivative norm, which
    sets the validity radius: the linearisation is good while
    |s - s0| << |h'| / |h''|.
    """
    h0 = np.asarray(jax.jit(h_of_s)(s0), float)
    d1 = np.asarray(jax.jit(jax.jacfwd(h_of_s))(s0), float)
    d2 = np.asarray(jax.jit(jax.jacfwd(jax.jacfwd(h_of_s)))(s0), float)
    P = np.eye(H) - np.ones((H, H)) / H if centred else np.eye(H)
    q = P @ d1
    c = P @ (h0 - s0 * d1)  # constant term of the tangent line
    nq = float(np.linalg.norm(q))
    if nq < 1e-13:
        return dict(n_q=nq, s_star=np.nan, r_star=np.nan, delta=np.nan,
                    q=q, c_perp=np.full(H, np.nan), curv=float(np.linalg.norm(d2)))
    s_star = -float(q @ c) / nq**2
    c_perp = c + s_star * q
    ncp = float(np.linalg.norm(c_perp))
    D = float(np.sqrt(ncp**2 + H * eps))
    curv = float(np.linalg.norm(d2))
    return dict(n_q=nq, s_star=s_star, r_star=D / nq, delta=D / np.sqrt(H),
                q=q, c_perp=c_perp, n_cperp=ncp, peak_dh=nq / D * np.sqrt(H),
                curv=curv, valid_radius=(nq / curv if curv > 0 else np.inf))


# --------------------------------------------------------------------------- #
# 4. geometry - encoder only                                                    #
# --------------------------------------------------------------------------- #


def profile_norm(g, s, H=16, eps=1e-6):
    """|dnhat/ds| in closed form, valid in BOTH regimes:

        |dnhat/ds| = (|q|/delta) * sqrt( [1 + (1-phi) t] / (1+t)^3 )
        t = ((s - x*) / r*)^2,   phi = H eps / (|c_perp|^2 + H eps)

    phi = 0 gives the Lorentzian (HWHM = r* exactly, peak x HWHM = sqrt(H));
    phi = 1 gives (1+t)^{-3/2}, whose HWHM/r* is sqrt(2^(2/3)-1) = 0.766421.
    A network initialised with zero encoder bias sits at phi = 1 exactly at
    u = 0, because c = 0 there, so the Lorentzian form is NOT what an untrained
    net shows. Verified against autodiff to 7e-13 across phi in [0,1].
    """
    phi = g["eps_frac"]
    t = ((np.asarray(s, float) - g["s_star"]) / g["r_star"]) ** 2
    return g["peak_dh"] * np.sqrt((1.0 + (1.0 - phi) * t) / (1.0 + t) ** 3)


def hwhm_ratio_pred(phi):
    """HWHM/r* predicted from phi alone. Positive real root of
       t^3 + 3 t^2 + (4 phi - 1) t - 3 = 0,  then HWHM/r* = sqrt(t)."""
    r = np.roots([1.0, 3.0, 4.0 * float(phi) - 1.0, -3.0])
    r = r[np.abs(r.imag) < 1e-12].real
    r = r[r > 0]
    return float(np.sqrt(r.min())) if len(r) else np.nan


def geometry(p, d, z0, H=16, eps=1e-6, centred=True):
    """Closed-form kink geometry for the sweep z(s) = z0 + s*d.

    d, z0 : (n_in,) numpy arrays.  Returns a dict of numpy scalars/vectors.
    """
    E = np.asarray(p["E"], float)
    b = np.asarray(p["b"], float)
    Wd = np.asarray(p["Wd"], float)[:, 0]
    d = np.asarray(d, float)
    z0 = np.asarray(z0, float)
    P = np.eye(H) - np.ones((H, H)) / H if centred else np.eye(H)

    q = P @ (E.T @ d)
    c = P @ (E.T @ z0 + b)
    nq = float(np.linalg.norm(q))
    a_skip = float((E.T @ d) @ Wd)  # exactly constant: the skip term
    if nq < 1e-13:
        return dict(a_skip=a_skip, n_q=nq, s_star=np.nan, r_star=np.nan,
                    delta=np.nan, q=q, c_perp=np.full(H, np.nan), n_cperp=np.nan,
                    peak_dh=np.nan, invariant=np.nan)
    s_star = -float(q @ c) / nq**2
    c_perp = c + s_star * q
    ncp = float(np.linalg.norm(c_perp))
    D = float(np.sqrt(ncp**2 + H * eps))
    delta = D / np.sqrt(H)
    r_star = D / nq
    return dict(a_skip=a_skip, n_q=nq, s_star=s_star, r_star=r_star, delta=delta,
                q=q, c_perp=c_perp, n_cperp=ncp, peak_dh=nq / delta,
                invariant=(nq / delta) * r_star, eps_frac=H * eps / (ncp**2 + H * eps))


def dnhat_closed(g, s, H=16, eps=1e-6):
    """Closed-form dnhat/ds on a grid of sweep offsets s (relative to z0)."""
    q, cp, ss = g["q"], g["c_perp"], g["s_star"]
    s = np.atleast_1d(np.asarray(s, float))
    r = np.outer(s - ss, q) + cp  # (n,H)
    D2 = (r**2).sum(1) + H * eps
    rq = r @ q
    return np.sqrt(H) * (q[None, :] / np.sqrt(D2)[:, None]
                         - r * (rq / D2**1.5)[:, None])


def lorentzian_norm(g, s, H=16):
    """|dnhat/ds| in the eps->0 limit: peak/(1 + ((s-s*)/r*)^2). HWHM = r*."""
    s = np.atleast_1d(np.asarray(s, float))
    t = (s - g["s_star"]) / g["r_star"]
    return g["peak_dh"] / (1.0 + t**2)


# --------------------------------------------------------------------------- #
# 5. measurement                                                                #
# --------------------------------------------------------------------------- #


def sweep_grid(g, X, n_uniform=20001, n_local=6001, span=60.0):
    """Uniform grid union a local grid resolving the kink. The uniform grid alone
    understates sup-error by ~2x (NOTES_direction1 5.3), so never use it alone."""
    xu = np.linspace(-X, X, n_uniform)
    if np.isfinite(g["r_star"]):
        loc = g["s_star"] + g["r_star"] * np.linspace(-span, span, n_local)
        loc = loc[(loc >= -X) & (loc <= X)]
        return np.unique(np.concatenate([xu, loc]))
    return xu


def common_grid(geoms, X, n_uniform=20001, n_local=6001, span=60.0):
    """Union of every arm's local grid. TV is monotone under refinement, so
    arms must be compared on ONE grid that resolves the sharpest of them."""
    parts = [np.linspace(-X, X, n_uniform)]
    for g in geoms:
        if np.isfinite(g.get("r_star", np.nan)):
            loc = g["s_star"] + g["r_star"] * np.linspace(-span, span, n_local)
            parts.append(loc[(loc >= -X) & (loc <= X)])
    return np.unique(np.concatenate(parts))


def measure(p, fwd, nhat, mvec, g, xs, d, z0, ctx=None, H=16, eps=1e-6,
            sigma0=1.0, gate=1.0):
    """Measured vs predicted Jacobian along the sweep. Returns numpy arrays."""
    d = jnp.asarray(d, float)
    z0 = jnp.asarray(z0, float)
    if ctx is None:
        ctx = jnp.zeros((0, H), dtype=jnp.float64)

    def F_of_s(s):
        return fwd(p, z0 + s * d, ctx, sigma0, gate)

    sj = jnp.asarray(xs)
    F = jax.vmap(F_of_s)(sj)
    J = jax.vmap(jax.grad(F_of_s))(sj)  # measured dF/ds, autodiff
    V = jax.vmap(lambda s: nhat(p, z0 + s * d, sigma0, gate))(sj)
    M = jax.vmap(lambda v: mvec(p, v, ctx))(V)
    dH_ad = jax.vmap(jax.jacfwd(lambda s: nhat(p, z0 + s * d, sigma0, gate)))(sj)

    M = np.asarray(M)
    dH_ad = np.asarray(dH_ad)
    dH_cf = dnhat_closed(g, xs, H, eps) if np.isfinite(g["r_star"]) else dH_ad
    return dict(
        s=np.asarray(xs), F=np.asarray(F), J=np.asarray(J),
        J_branch=np.asarray(J) - g["a_skip"],
        J_pred=g["a_skip"] + np.einsum("nh,nh->n", M, dH_cf),  # closed form x VJP
        J_pred_ad=g["a_skip"] + np.einsum("nh,nh->n", M, dH_ad),  # VJP x autodiff
        m=M, dH_cf=dH_cf, dH_ad=dH_ad,
        norm_dH_cf=np.linalg.norm(dH_cf, axis=1),
        norm_dH_ad=np.linalg.norm(dH_ad, axis=1),
    )


def hwhm(s, y):
    """Half-width at half-maximum of a single-peaked curve, linear interpolation."""
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    i = int(np.argmax(y))
    half = 0.5 * y[i]
    out = []
    for side in (-1, +1):
        j = i
        while 0 <= j + side < len(y) and y[j + side] > half:
            j += side
        k = j + side
        if k < 0 or k >= len(y):
            return np.nan
        t = (y[j] - half) / (y[j] - y[k])
        out.append(abs(s[j] + t * (s[k] - s[j]) - s[i]))
    return float(np.mean(out))


def total_variation(s, J):
    """TV of the measured Jacobian. No ground-truth derivative is used."""
    return float(np.abs(np.diff(np.asarray(J))).sum())


PREDICTABLE = {"layer": True, "rms": True, "taper": False, "static": False, "none": False}


def evaluate(p, cfg, d, z0, ctx=None, X=4.0, sigma0=1.0, gate=1.0, grid=None):
    """E0 in one call: geometry, measurement, amplitude, width, TV, self-tests.

    The closed form is asserted only for `layer` and `rms`. For `static`,
    `none`, `taper` and `postnorm` the geometry is MEASURED and the predicted
    columns are NaN: those arms either have no closest-approach point or a
    denominator Theorem 1 does not cover. Reporting NaN there is the point of
    the arm, not a gap.
    """
    H, eps = cfg["H"], cfg["eps"]
    kind_norm, kind_branch = cfg["norm"], cfg["branch"]
    postnorm = cfg.get("postnorm", False)
    predictable = PREDICTABLE[kind_norm] and not postnorm
    fwd, nhat, mvec = make_block(kind_norm, kind_branch, H, eps,
                                 heads=cfg.get("heads", 4), postnorm=postnorm)
    g = geometry(p, d, z0, H, eps, centred=CENTRED[kind_norm])
    if not predictable:
        g = dict(g, s_star=np.nan, r_star=np.nan, delta=np.nan, peak_dh=np.nan,
                 invariant=np.nan, n_cperp=np.nan, eps_frac=np.nan)
        if postnorm:
            g["a_skip"] = np.nan
    xs = (np.asarray(grid, float) if grid is not None
          else sweep_grid(g, X, cfg.get("n_uniform", 20001), cfg.get("n_local", 6001)))
    r = measure(p, fwd, nhat, mvec, g, xs, d, z0, ctx, H, eps, sigma0, gate)

    out = dict(g)
    out.pop("q"), out.pop("c_perp")
    out["norm"], out["branch"], out["predictable"] = kind_norm, kind_branch, predictable
    # --- self-tests (section 9.1): identities that must hold to ~1e-12 --------
    if predictable:
        out["test_dnhat_max_abs"] = float(np.abs(r["dH_cf"] - r["dH_ad"]).max())
        out["test_J_closed_max_abs"] = float(np.abs(r["J"] - r["J_pred"]).max())
        out["test_cperp_orth"] = float(abs(g["c_perp"] @ g["q"]))
    else:
        out["test_dnhat_max_abs"] = out["test_J_closed_max_abs"] = np.nan
        out["test_cperp_orth"] = np.nan
    # the decomposition itself holds for every pre-norm arm, closed form or not
    out["test_J_decomp_max_abs"] = (np.nan if postnorm
                                    else float(np.abs(r["J"] - r["J_pred_ad"]).max()))

    # --- amplitude at the closest-approach point (Proposition 3) -------------
    if np.isfinite(g["r_star"]) and -X <= g["s_star"] <= X:
        # evaluate exactly AT the closest-approach point, not at a grid neighbour
        rs = measure(p, fwd, nhat, mvec, g, np.array([g["s_star"]]), d, z0, ctx,
                     H, eps, sigma0, gate)
        m_star = rs["m"][0]
        out["amp_pred"] = float(m_star @ g["q"] / g["delta"])
        out["amp_meas"] = float(rs["J_branch"][0])
        out["amp_relerr"] = abs(out["amp_pred"] - out["amp_meas"]) / max(abs(out["amp_meas"]), 1e-300)
        out["amp_sf"] = -np.log10(max(out["amp_relerr"], 1e-17))
        # the claim quoted in the paper: closed form at x* vs the measured PEAK of
        # |J_branch|, which sits slightly off x* because m(x) is not constant
        ipk = int(np.argmax(np.abs(r["J_branch"])))
        out["amp_peak_meas"] = float(r["J_branch"][ipk])
        out["amp_peak_relerr"] = abs(out["amp_pred"] - out["amp_peak_meas"]) / max(abs(out["amp_peak_meas"]), 1e-300)
        out["amp_peak_sf"] = -np.log10(max(out["amp_peak_relerr"], 1e-17))
        out["peak_offset_over_rstar"] = float((xs[ipk] - g["s_star"]) / g["r_star"])
        out["cos_mq"] = float(m_star @ g["q"] / (np.linalg.norm(m_star) * g["n_q"]))
        out["norm_m_star"] = float(np.linalg.norm(m_star))
        # constant-m baseline: how much of the amplitude survives WITHOUT the
        # pointwise VJP?  m frozen at the far field.  If this is close, the
        # pointwise evaluation is not doing the work; if it is far, say so.
        m_far = r["m"][int(np.argmax(np.abs(xs - g["s_star"])))]
        out["amp_pred_constm"] = float(m_far @ g["q"] / g["delta"])
        out["amp_constm_relerr"] = abs(out["amp_pred_constm"] - out["amp_meas"]) / max(abs(out["amp_meas"]), 1e-300)

        hw = hwhm(xs, r["norm_dH_ad"])
        out["hwhm_meas"] = hw
        out["hwhm_over_rstar"] = hw / g["r_star"]
        # phi-corrected prediction: HWHM = r* only in the geometry regime
        out["phi"] = g["eps_frac"]
        out["hwhm_over_rstar_pred"] = hwhm_ratio_pred(g["eps_frac"])
        out["hwhm_ratio_relerr"] = abs(out["hwhm_over_rstar"]
                                       / out["hwhm_over_rstar_pred"] - 1.0)
        out["invariant_pred"] = np.sqrt(H) * out["hwhm_over_rstar_pred"]
        out["profile_max_abs_err"] = float(
            np.abs(profile_norm(g, xs, H, eps) - r["norm_dH_ad"]).max())
        # width of the Jacobian lobe itself (branch-sensitive, unlike |dnhat/ds|)
        hwj = hwhm(xs, np.abs(r["J_branch"]))
        out["hwhm_J_over_rstar"] = hwj / g["r_star"]
        out["peak_dh_meas"] = float(r["norm_dH_ad"].max())
        out["peak_over_pred"] = out["peak_dh_meas"] / g["peak_dh"]
        out["invariant_meas"] = out["peak_dh_meas"] * hw
    else:
        for k in ("amp_pred", "amp_meas", "amp_relerr", "amp_sf", "amp_peak_meas",
                  "amp_peak_relerr", "amp_peak_sf", "peak_offset_over_rstar",
                  "amp_pred_constm", "amp_constm_relerr", "hwhm_J_over_rstar",
                  "phi", "hwhm_over_rstar_pred", "hwhm_ratio_relerr",
                  "invariant_pred", "profile_max_abs_err",
                  "cos_mq", "norm_m_star", "hwhm_meas", "hwhm_over_rstar",
                  "peak_dh_meas", "peak_over_pred", "invariant_meas"):
            out[k] = np.nan

    inside = np.abs(xs) <= X
    out["TV"] = total_variation(xs[inside], r["J"][inside])
    out["TV_uniform_only"] = total_variation(
        np.linspace(-X, X, cfg.get("n_uniform", 20001)),
        np.interp(np.linspace(-X, X, cfg.get("n_uniform", 20001)), xs, r["J"]))
    out["n_grid"] = int(inside.sum())
    out["J_sup"] = float(np.abs(r["J"][inside]).max())
    out["J_med"] = float(np.median(r["J"][inside]))
    # arm-agnostic: needs neither a_skip nor a ground-truth derivative
    out["J_dev_sup"] = float(np.abs(r["J"][inside] - out["J_med"]).max())
    out["J_branch_sup"] = (np.nan if not np.isfinite(g["a_skip"])
                           else float(np.abs(r["J_branch"][inside]).max()))
    out["L1_branch"] = (np.nan if not np.isfinite(g["a_skip"])
                        else float(np.trapezoid(np.abs(r["J_branch"][inside]), xs[inside])))
    return out, r


# --------------------------------------------------------------------------- #
# 6. plants and data                                                            #
# --------------------------------------------------------------------------- #


def plant_linear(x, u, a=1.03, bb=0.01):
    return a * x + bb * u


def plant_sinc(x, u, a=1.03, bb=0.01, amp=0.5, w=3.0):
    return a * x + bb * u + amp * np.sinc(w * x / np.pi)


PLANTS = {"linear": plant_linear, "sinc": plant_sinc}


def make_data_mimo(n=20000, X=4.0, U=10.0, seed=0, dt=0.01,
                   A=((0.0, 1.0), (-2.0, -0.5)), B=(0.0, 1.0)):
    """ZOH discretisation of xdot = A x + B u, exactly as E2 specifies.

    Returns z = (x1, x2, u) and y = the FIRST state at the next step, so the
    block stays scalar-output and every other piece of the harness is unchanged.
    Ad, Bd are returned for the skip-matching reference.
    """
    from scipy.linalg import expm
    A = np.asarray(A, float)
    B = np.asarray(B, float).reshape(2, 1)
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2:] = B
    E = expm(M * dt)
    Ad, Bd = E[:2, :2], E[:2, 2:]
    r = np.random.default_rng(seed)
    x = r.uniform(-X, X, (n, 2))
    u = r.uniform(-U, U, (n, 1))
    y = (x @ Ad.T + u @ Bd.T)[:, 0]
    return dict(z=np.concatenate([x, u], 1), y=y, ctx=None, Ad=Ad, Bd=Bd, dt=dt)


def unit_directions(k=8, n_state=2):
    """k unit directions in state space, padded with a zero input component."""
    th = np.linspace(0.0, np.pi, k, endpoint=False)
    d = np.stack([np.cos(th), np.sin(th)], 1)
    return np.concatenate([d, np.zeros((k, 1))], 1), th


def make_data(plant="linear", n=20000, X=4.0, U=10.0, seed=0, L=1, a=1.03, bb=0.01):
    """i.i.d. (x,u) pairs for L=1; for L>1 a genuine backward plant rollout whose
    LAST state is exactly U(-X,X), so every arm sees the same marginal in x."""
    r = np.random.default_rng(seed)
    x = r.uniform(-X, X, n)
    u = r.uniform(-U, U, n)
    y = PLANTS[plant](x, u)
    if L == 1:
        return dict(z=np.stack([x, u], 1), y=y, ctx=None)
    ctx = np.zeros((n, L - 1, 2))
    xk = x.copy()
    for j in range(L - 1):  # walk backwards: x_{k-1} = (x_k - bb u_{k-1}) / a
        up = r.uniform(-U, U, n)
        xp = (xk - bb * up) / a
        ctx[:, L - 2 - j, 0] = xp
        ctx[:, L - 2 - j, 1] = up
        xk = xp
    return dict(z=np.stack([x, u], 1), y=y, ctx=ctx)


# --------------------------------------------------------------------------- #
# 7. training                                                                   #
# --------------------------------------------------------------------------- #


def train(p, data, cfg, extra_loss=None, sigma0=1.0, gate=1.0, verbose=True):
    """Adam via lax.scan in chunks. extra_loss(p, key) -> scalar is added to MSE."""
    H, eps = cfg["H"], cfg["eps"]
    fwd, nhat, mvec = make_block(cfg["norm"], cfg["branch"], H, eps,
                                 heads=cfg.get("heads", 4),
                                 postnorm=cfg.get("postnorm", False))
    Z = jnp.asarray(data["z"])
    Y = jnp.asarray(data["y"])
    if data["ctx"] is None:
        CTX = None
        batched = jax.vmap(lambda p_, z_: fwd(p_, z_, None, sigma0, gate), in_axes=(None, 0))
    else:
        CTX = jnp.asarray(data["ctx"])

        def batched(p_, z_, c_, enc_=None):
            def one(zz, cc):
                ctok = norm_apply(cfg["norm"], enc(p_, cc), eps, sigma0, gate)
                return fwd(p_, zz, ctok, sigma0, gate)
            return jax.vmap(one)(z_, c_)

    b1, b2, aeps = cfg.get("adam", (0.9, 0.999, 1e-8))
    lr = cfg.get("lr", 3e-3)
    steps, chunk = cfg.get("steps", 3000), cfg.get("chunk", 500)

    def loss(pp):
        pred = batched(pp, Z) if CTX is None else batched(pp, Z, CTX)
        base = jnp.mean((pred - Y) ** 2)
        return base if extra_loss is None else base + extra_loss(pp)

    @jax.jit
    def run_chunk(p, m, v, t0):
        def st(carry, i):
            p, m, v = carry
            t = t0 + i + 1.0
            gr = jax.grad(loss)(p)
            m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, gr)
            v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, v, gr)
            mh = jax.tree.map(lambda a: a / (1 - b1**t), m)
            vh = jax.tree.map(lambda a: a / (1 - b2**t), v)
            p = jax.tree.map(lambda a, b, c: a - lr * b / (jnp.sqrt(c) + aeps), p, mh, vh)
            return (p, m, v), 0.0
        (p, m, v), _ = jax.lax.scan(st, (p, m, v), jnp.arange(chunk, dtype=jnp.float64))
        return p, m, v

    m = jax.tree.map(jnp.zeros_like, p)
    v = jax.tree.map(jnp.zeros_like, p)
    t0 = time.time()
    hist = []
    for ci in range(steps // chunk):
        p, m, v = run_chunk(p, m, v, float(ci * chunk))
        mse = float(loss(p))
        hist.append(dict(step=(ci + 1) * chunk, loss=mse))
    if verbose:
        print(f"    loss {hist[-1]['loss']:.6e}  [{time.time() - t0:.0f}s]")
    return p, hist


def frozen_sigma(p, data, cfg):
    """sigma0 for the static/taper arms: LN scale at init, averaged over the data."""
    h = np.asarray(enc(p, jnp.asarray(data["z"])))
    r = h - h.mean(-1, keepdims=True)
    return float(np.sqrt((r**2).mean(-1) + cfg["eps"]).mean())


def var_ratio(data):
    return float(np.var(data["y"]))


# --------------------------------------------------------------------------- #
# 8. results directory                                                          #
# --------------------------------------------------------------------------- #


def outdir(tag):
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    d = f"{base}/kink_out/{tag}"
    for sub in ("fig", "ckpt", "arrays"):
        os.makedirs(f"{d}/{sub}", exist_ok=True)
    return d


def save_ckpt(path, p, meta):
    flat = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}{k}/", v)
        else:
            flat[prefix[:-1]] = np.asarray(node)
    walk("", p)
    np.savez(path, meta_json=np.array(json.dumps(meta)), **flat)


def env_stamp():
    import scipy
    return dict(jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__,
                x64=bool(jax.config.read("jax_enable_x64")),
                devices=[str(x) for x in jax.devices()])
