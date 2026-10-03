"""R6 criterion stage: the pure computations, without files, torch or checkpoints, so
that tests can exercise them on synthetic arrays. numpy float64 and JAX float64.

Every definition is quoted from the tagged pre-registration files, which are its
only source (CLAUDE.md): prereg/r6.md (tag prereg-r6), r6-deviations.md (D1-D4,
tag prereg-r6-d1), r6-deviations-2.md (D5, prereg-r6-d2) and r6-deviations-3.md (D6,
prereg-r6-d3). Lens quantities come from lens/geometry.py (docs/theory.md).

r6.md, Definitions:
    "μ, C: sample mean and covariance of all observations of a task (numpy.cov
    conventions, ddof = 1).
    Kept dimensions: eigenvectors of C whose eigenvalue is at least 1e-10 times the
    largest. m is their number. C⁺ is the pseudo-inverse of C restricted to them.
    d₁: the unit eigenvector of C with the largest eigenvalue.
    D: project the observations onto d₁, p = (z − μ)·d₁, and set
    D = (q97.5(p) − q2.5(p)) / 2, with numpy's default (linear) percentile method.
    Width along d₁: r_eff(d₁) = r*(d₁)·√(1 + κ) (convention 6). For a degenerate lens
    this is the ε-limited width (convention 4).
    Whitened distance: w(x, y) = √((x − y)ᵀ C⁺ (x − y)).
    Lens distance: ρ(z) = ‖A(z − z*)‖² / ‖c⊥‖² as in docs/theory.md (convention 2),
    defined only for a non-degenerate lens. Effective lens distance:
    ρ_eff(z) = ‖A(z − z*)‖² / (‖c⊥‖² + Hε)"

r6.md, Criterion:
    "A task's lens is "sharp and in the data" if all three hold.
    1. Inside: w(z*, μ)² is at most the 95% quantile of χ² with m degrees of freedom.
    2. Populated: draw a subsample of 5,000 observations with
       numpy.random.default_rng(0).choice(n, 5000, replace=False), where n is the
       task's number of observations. For each subsample observation, compute the
       whitened distance to its nearest *other* subsample observation. The whitened
       distance from z* to its nearest subsample observation must be at most the 95th
       percentile of those distances.
    3. Sharp: D / r_eff(d₁) ≥ 5."

r6.md, Gate G1:
    "G1 passes if the criterion holds for at least 2 of the 4 tasks other than
    dog-run." Seed 1 counts (r6.md, Models: "only seed 1 counts toward the gate").

D6 (a): "If this happens [planner return below 0.5 of published] to the checkpoint
    that counts toward G1 for a task (seed 1, or seed 2 for cartpole-swingup if seed 1
    ends up unidentified under (b)), that task leaves G1, and G1 then needs 2 of the
    remaining non-dog tasks."

D6 (b), condition (i), (ii) and the humanoid-run seed 3 label: see identification().
D6 (g): see susceptibility().
"""

import dataclasses
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def lens_geometry():
    """lens/geometry.py, loaded by path (the lens package's __init__ imports core)."""
    spec = importlib.util.spec_from_file_location(
        "lens_geometry", os.path.join(ROOT, "lens", "geometry.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


geo = lens_geometry()


# --------------------------------------------------------------------------- #
# r6.md Definitions                                                             #
# --------------------------------------------------------------------------- #


def canonical_sign(v):
    """Eigenvectors are defined up to sign. Fix it so that the component of largest
    magnitude is positive (the first such component on a tie). Affects only the sign
    of signed outputs such as the t of a peak; every criterion quantity is
    sign-invariant."""
    v = np.asarray(v, np.float64)
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


@dataclasses.dataclass(frozen=True)
class DataStats:
    n: int
    k: int
    mu: np.ndarray            # (k,)
    C: np.ndarray             # (k, k), numpy.cov, ddof = 1
    eigvals: np.ndarray       # (k,), descending
    eigvecs: np.ndarray       # (k, k), columns, same order, canonical sign
    kept: np.ndarray          # (k,) bool: eigenvalue >= kept_rtol * largest
    m: int
    whitener: np.ndarray      # (k, m): w(x, y) = ||(x - y) @ whitener||
    C_plus: np.ndarray        # (k, k): pseudo-inverse of C restricted to the kept ones
    d1: np.ndarray
    d2: np.ndarray
    d2_kept: bool             # D6 (g): d2 degenerate unless its eigenvalue is kept


def data_stats(X, kept_rtol=1e-10):
    """μ, C, kept dimensions, m, C⁺, d₁ (r6.md) and d₂ (D6 (g)) of observations X
    (n, k) in layer-input coordinates. d₁ and d₂ are eigenvectors of the full C: r6.md
    has no kept-dimension step for d₁ (D6 (g), "Kept dimensions")."""
    X = np.asarray(X, np.float64)
    n, k = X.shape
    mu = X.mean(axis=0)
    C = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    V = np.stack([canonical_sign(V[:, i]) for i in range(k)], axis=1)
    kept = w >= kept_rtol * w[0]
    Vk, wk = V[:, kept], w[kept]
    return DataStats(n=n, k=k, mu=mu, C=C, eigvals=w, eigvecs=V, kept=kept,
                     m=int(kept.sum()), whitener=Vk / np.sqrt(wk),
                     C_plus=(Vk / wk) @ Vk.T, d1=V[:, 0], d2=V[:, 1] if k > 1 else None,
                     d2_kept=bool(k > 1 and kept[1]))


def whitened(S, x, y):
    """w(x, y) = √((x − y)ᵀ C⁺ (x − y)), for x (..., k) and y (k,) or (..., k)."""
    return np.linalg.norm((np.asarray(x, np.float64) - y) @ S.whitener, axis=-1)


def half_width_D(S, X, q=(2.5, 97.5)):
    """D = (q97.5(p) − q2.5(p)) / 2, p = (z − μ)·d₁, numpy's default (linear) method."""
    p = (np.asarray(X, np.float64) - S.mu) @ S.d1
    lo, hi = np.percentile(p, q)
    return float((hi - lo) / 2.0)


# --------------------------------------------------------------------------- #
# r6.md Criterion                                                               #
# --------------------------------------------------------------------------- #


def inside(S, z_star, quantile=0.95):
    """Inside: w(z*, μ)² ≤ the `quantile` quantile of χ² with m degrees of freedom.
    Also reported: the part of z* − μ outside the kept eigenvectors, which C⁺ ignores."""
    from scipy.stats import chi2
    dz = np.asarray(z_star, np.float64) - S.mu
    w2 = float(whitened(S, z_star, S.mu) ** 2)
    thr = float(chi2.ppf(quantile, S.m))
    Vk = S.eigvecs[:, S.kept]
    outside = float(np.linalg.norm(dz - Vk @ (Vk.T @ dz)))
    return dict(inside_w2=w2, inside_chi2_q=thr, inside=bool(w2 <= thr),
                inside_m=S.m, inside_norm_outside_kept=outside)


def nearest_other(Y, chunk=256):
    """For each row of Y (whitened coordinates), the Euclidean distance to its nearest
    other row (another index; an exact duplicate at another index counts, at 0)."""
    from scipy.spatial.distance import cdist
    n = len(Y)
    out = np.empty(n)
    for i in range(0, n, chunk):
        Dm = cdist(Y[i:i + chunk], Y)  # direct differences, not the |a|^2 + |b|^2 - 2ab form
        Dm[np.arange(len(Dm)), np.arange(i, i + len(Dm))] = np.inf
        out[i:i + chunk] = Dm.min(axis=1)
    return out


def populated(S, X, z_star, n_sub=5000, rng_seed=0, percentile=95.0):
    """Populated (r6.md, quoted in the module docstring). The subsample indices are
    numpy.random.default_rng(rng_seed).choice(n, n_sub, replace=False) over the rows of
    X in their stored order (episode-major, D6 (e)); the percentile is numpy's default
    (linear) method."""
    from scipy.spatial.distance import cdist
    X = np.asarray(X, np.float64)
    idx = np.random.default_rng(rng_seed).choice(len(X), n_sub, replace=False)
    Y = (X[idx] - S.mu) @ S.whitener
    nn = nearest_other(Y)
    thr = float(np.percentile(nn, percentile))
    zw = ((np.asarray(z_star, np.float64) - S.mu) @ S.whitener)[None]
    dz = cdist(zw, Y)[0]
    j = int(np.argmin(dz))
    return dict(populated_dist=float(dz[j]), populated_threshold=thr,
                populated=bool(dz[j] <= thr), populated_nearest_index=int(idx[j]),
                populated_nn_median=float(np.median(nn)), populated_n=int(n_sub)), idx


def sharp(S, X, L, minimum=5.0):
    """Sharp: D / r_eff(d₁) ≥ 5, with r_eff along d₁ through z* (geometry.line)."""
    D = half_width_D(S, X)
    ln = geo.line(L, S.d1)
    ratio = D / ln["r_eff"]
    return dict(D=D, r_star_d1=float(ln["r_star"]), r_eff_d1=float(ln["r_eff"]),
                sharp_ratio=float(ratio), sharp=bool(ratio >= minimum))


def rho_fractions(L, X, levels=(1, 10, 100, 1000)):
    """r6.md, reported regardless: the fractions of observations with ρ_eff ≤ each
    level, and the same for ρ when the lens is non-degenerate."""
    dz = np.asarray(X, np.float64) - L.z_star
    Adz2 = np.sum((dz @ L.A.T) ** 2, axis=-1)
    rho_eff = Adz2 / (L.norm_c_perp ** 2 + L.H * L.eps)
    out = {f"frac_rho_eff_le_{lv:g}": float(np.mean(rho_eff <= lv)) for lv in levels}
    if not L.degenerate:
        rho = Adz2 / L.norm_c_perp ** 2
        out.update({f"frac_rho_le_{lv:g}": float(np.mean(rho <= lv)) for lv in levels})
    out["rho_eff_median"] = float(np.median(rho_eff))
    return out


def criterion(S, X, L, cfg_c):
    """All three criterion quantities and the conjunction, for one checkpoint."""
    out = {}
    out.update(inside(S, L.z_star, float(cfg_c["inside_quantile"])))
    pop, idx = populated(S, X, L.z_star, int(cfg_c["populated"]["n_sub"]),
                         int(cfg_c["populated"]["rng_seed"]),
                         float(cfg_c["populated"]["percentile"]))
    out.update(pop)
    out.update(sharp(S, X, L, float(cfg_c["sharp_min"])))
    out["criterion"] = bool(out["inside"] and out["populated"] and out["sharp"])
    return out, idx


# --------------------------------------------------------------------------- #
# D6 (b): identification; G1                                                    #
# --------------------------------------------------------------------------- #


def condition_ii(err, candidate="symlog", e_over_e0_max=0.1):
    """D6 (b) (ii): "symlog gives the lowest e among the three D3 candidates, with
    e/e₀ ≤ 0.1 (D3's computation)". err: {candidate: dict(e, e0)} from
    collect.consistency_errors; e₀ is the candidate's own (as consistency.csv's
    e_over_e0). "Lowest": e_symlog ≤ every other e (as collect.decide)."""
    e = err[candidate]["e"]
    lowest = all(e <= err[c]["e"] for c in err)
    ratio = e / err[candidate]["e0"]
    return dict(lowest=bool(lowest), e_over_e0=float(ratio),
                holds=bool(lowest and ratio <= e_over_e0_max),
                within_1p1=sorted(c for c in err if c != candidate and err[c]["e"] <= 1.1 * e))


def identification(fraction_gate, cii_gate, fraction_rep, cii_rep, return_ratio=0.9):
    """D6 (b).

    cartpole-swingup seed 1: "(i) planner return ≥ 0.9 × published under
    eval_mode=True [...] the return being the mean over the checkpoint's 50 episodes
    [...] AND (ii) [...]. If either condition fails, cartpole-swingup seed 1 is
    unidentified and cartpole-swingup's G1 vote uses seed 2."

    humanoid-run seed 3: ""confirmed" if condition (i) holds (planner return as the
    mean over its 50 episodes) and symlog gives the lowest e on the planner data;
    "identified by consistency" if symlog gives the lowest e on the planner data but
    condition (i) fails [...]; "unidentified" if symlog does not give the lowest e
    on the planner data." The 0.1 threshold is not applied to humanoid-run.
    """
    i_gate = bool(fraction_gate >= return_ratio)
    gate = dict(condition_i=i_gate, fraction=float(fraction_gate),
                condition_ii=bool(cii_gate["holds"]), identified=bool(i_gate and cii_gate["holds"]))
    i_rep = bool(fraction_rep >= return_ratio)
    if not cii_rep["lowest"]:
        label = "unidentified"
    elif i_rep:
        label = "confirmed"
    else:
        label = "identified by consistency"
    rep = dict(condition_i=i_rep, fraction=float(fraction_rep), symlog_lowest=bool(cii_rep["lowest"]),
               label=label)
    return gate, rep


def g1_vote(results, cartpole_s1_identified, flags, tasks, min_pass=2,
            counting_seed=1, fallback=("cartpole-swingup", 2)):
    """r6.md G1 with D6 (a)'s flag rule and D6 (b)'s cartpole-swingup fallback.

    results: {(task, seed): criterion bool}; flags: {(task, seed): below 0.5 x published}.
    tasks: the 4 tasks other than dog-run. A task whose counting checkpoint is flagged
    leaves G1; G1 then needs min_pass of the remaining tasks.
    """
    rows, passes = [], 0
    for t in tasks:
        seed = counting_seed
        why = "seed 1"
        if t == fallback[0] and not cartpole_s1_identified:
            seed, why = fallback[1], "seed 1 unidentified under D6 (b): seed 2"
        k = (t, seed)
        left = bool(flags.get(k, False))
        holds = bool(results[k])
        rows.append(dict(task=t, counting_seed=seed, why=why, flagged_below_half=left,
                         counts=not left, criterion=holds))
        passes += int(holds and not left)
    remaining = sum(r["counts"] for r in rows)
    return dict(rows=rows, n_counting=remaining, n_pass=passes, min_pass=min_pass,
                g1=bool(passes >= min_pass))


# --------------------------------------------------------------------------- #
# networks in JAX (for D6 (g)'s Jacobian-vector products)                       #
# --------------------------------------------------------------------------- #


def _path_key(p):
    return tuple(int(x) for x in p.split("."))


def mlp_spec(sd, layout, prefix, final):
    """The ops of one network as (kind, W, b) with kind 'linear' or 'ln', and an
    activation after each 'ln' ('mish' or 'simnorm'), in the order layouts.py
    evaluates them. Call layouts.build_networks first: it checks the layout and stops
    on anything unexpected; this only re-reads the same keys."""
    ops = []
    if layout == "public":
        idx = sorted({int(k[len(prefix) + 1:].split(".")[0]) for k in sd if k.startswith(prefix + ".")})
        for i in idx:
            ops.append(("linear", sd[f"{prefix}.{i}.weight"], sd[f"{prefix}.{i}.bias"], None))
            if f"{prefix}.{i}.ln.weight" in sd:
                act = "simnorm" if (i == idx[-1] and final == "simnorm") else "mish"
                ops.append(("ln", sd[f"{prefix}.{i}.ln.weight"], sd[f"{prefix}.{i}.ln.bias"], act))
    elif layout == "prerelease":
        paths = sorted({k[len(prefix) + 1:].rsplit(".", 1)[0] for k in sd if k.startswith(prefix + ".")},
                       key=_path_key)
        for j, p in enumerate(paths):
            w, b = sd[f"{prefix}.{p}.weight"], sd[f"{prefix}.{p}.bias"]
            if np.ndim(w) == 2:
                ops.append(("linear", w, b, None))
            else:
                act = "simnorm" if (j == len(paths) - 1 and final == "simnorm") else "mish"
                ops.append(("ln", w, b, act))
    else:
        raise ValueError(layout)
    return ops


def jax_mlp(ops, eps, simnorm_dim):
    """f(x) in JAX float64 for ops from mlp_spec; x in layer-input coordinates."""
    import jax.numpy as jnp
    P = [(k, jnp.asarray(w, jnp.float64), jnp.asarray(b, jnp.float64), a) for k, w, b, a in ops]

    def f(x):
        for kind, w, b, act in P:
            if kind == "linear":
                x = x @ w.T + b
                continue
            m = x.mean(-1, keepdims=True)
            v = ((x - m) ** 2).mean(-1, keepdims=True)
            x = (x - m) / jnp.sqrt(v + eps) * w + b
            if act == "mish":
                x = x * jnp.tanh(jnp.logaddexp(0.0, x))
            else:
                s = x.shape
                y = x.reshape(*s[:-1], -1, simnorm_dim)
                y = jnp.exp(y - y.max(-1, keepdims=True))
                x = (y / y.sum(-1, keepdims=True)).reshape(s)
        return x
    return f


# --------------------------------------------------------------------------- #
# D6 (g): end-to-end susceptibility                                             #
# --------------------------------------------------------------------------- #


def ln_hat(E, b, eps, X):
    """Layer-1 LayerNorm output before γ and β (theory.md's ĥ) at inputs X (..., k)."""
    h = np.asarray(X, np.float64) @ np.asarray(E, np.float64).T + b
    r = h - h.mean(-1, keepdims=True)
    return r / np.sqrt((r ** 2).mean(-1, keepdims=True) + eps)


def corollary1(L, d, t):
    """D6 (g): ĥ(t) = √H (cos θ ĉ + sin θ q̂) / √(1 + κ cos² θ), θ = arctan(t / r*(d)),
    q̂ = A d / ‖A d‖, along x(t) = z* + t d (non-degenerate lens)."""
    q = L.A @ d
    qh = q / np.linalg.norm(q)
    th = np.arctan(np.asarray(t, np.float64) / geo.line(L, d)["r_star"])[:, None]
    return np.sqrt(L.H) * (np.cos(th) * L.c_hat + np.sin(th) * qh) / np.sqrt(1.0 + L.kappa * np.cos(th) ** 2)


def theta_grid(L, d, X, n_grid=2001, t_range=3.0):
    """D6 (g) Lines: 2,001 points uniform in θ over |θ| ≤ arctan(T / r*(d)), with
    T = max(max_i |⟨x_i − z*, d⟩|, 3·r_eff(d))."""
    ln = geo.line(L, d)
    T = max(float(np.max(np.abs((np.asarray(X, np.float64) - L.z_star) @ d))), t_range * ln["r_eff"])
    th = np.linspace(-np.arctan(T / ln["r_star"]), np.arctan(T / ln["r_star"]), n_grid)
    return ln["r_star"] * np.tan(th), dict(T=T, r_star=ln["r_star"], r_eff=ln["r_eff"])


def jvp_norms(f, X, d, batch=4096):
    """‖J_f(x) d‖ for each row x of X, by forward-mode autodiff (jax.jvp)."""
    import jax
    import jax.numpy as jnp
    dj = jnp.asarray(d, jnp.float64)
    one = jax.jit(jax.vmap(lambda x: jax.jvp(f, (x,), (dj,))[1]))
    out = []
    for i in range(0, len(X), batch):
        out.append(np.linalg.norm(np.asarray(one(jnp.asarray(X[i:i + batch], jnp.float64))), axis=-1))
    return np.concatenate(out)


def susceptibility(L, E, b, d, X, outputs, n_grid=2001, t_range=3.0, cor1_rtol=1e-10):
    """D6 (g) for one direction d (unit, layer-input coordinates). E, b: layer 1.

    outputs: {name: (f_line, {key: (f_data, rows, direction)})}. f_line maps a layer
    input (k,) to the output, with the action fixed at a = 0 for the dynamics; it is
    evaluated along the line. Each data set gives a function of one row, the rows
    and the direction to differentiate along: for the a = 0 version the rows are
    the states x_i and the direction d; for the recorded-action version the rows are
    [x_i, a_i] and the direction [d, 0], so the action is held fixed.
    Returns the Corollary 1 check and, per output: ‖g′(0)‖, S = ‖g′(0)‖ / r*(d),
    ‖g′(0)‖ / r_eff(d), the peak ‖J(t)‖ over grid points with |t| ≤ 3·r_eff(d) and
    its t, the medians over data states, and the ratio of the peak to the median.
    """
    d = np.asarray(d, np.float64)
    t, info = theta_grid(L, d, X, n_grid, t_range)
    line = L.z_star + t[:, None] * d
    hh = ln_hat(E, b, L.eps, line)
    cf = corollary1(L, d, t)
    dev = float(np.max(np.linalg.norm(hh - cf, axis=-1) / np.linalg.norm(cf, axis=-1)))
    res = dict(T=info["T"], r_star=info["r_star"], r_eff=info["r_eff"], n_grid=n_grid,
               cor1_max_rel_dev=dev, cor1_ok=bool(dev <= cor1_rtol), out={})
    rs = info["r_star"]
    near = np.abs(t) <= t_range * info["r_eff"]
    for name, (f_line, data_sets) in outputs.items():
        Jn = jvp_norms(f_line, line, d)
        J0 = float(jvp_norms(f_line, L.z_star[None], d)[0])
        g0 = J0 * rs  # g′(0) = J(0) (r*² + 0) / r*
        gp = Jn * (rs ** 2 + t ** 2) / rs
        i = int(np.argmax(np.where(near, Jn, -np.inf)))
        o = dict(g_prime_0=g0, S=g0 / rs, S_eff=g0 / info["r_eff"], peak_J=float(Jn[i]),
                 t_peak=float(t[i]), t_peak_over_r_eff=float(t[i] / info["r_eff"]),
                 J_line=Jn, g_prime_line=gp)
        for key, (f_d, rows, dvec) in data_sets.items():
            o[f"median_J_{key}"] = float(np.median(jvp_norms(f_d, rows, dvec)))
            o[f"n_{key}"] = int(len(rows))
        o["peak_over_median_a0"] = o["peak_J"] / o["median_J_a0"]
        res["out"][name] = o
    res["t"] = t
    return res
