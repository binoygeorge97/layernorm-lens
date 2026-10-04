"""R6 criterion stage: the pure computations, without files, torch or checkpoints, so
that tests can exercise them on synthetic arrays. numpy float64 and JAX float64.

Every rule is quoted from the tagged pre-registration files, which are its only
source (CLAUDE.md): prereg/r6.md (tag prereg-r6), r6-deviations.md (D1-D4, tag
prereg-r6-d1), r6-deviations-2.md (D5, prereg-r6-d2), r6-deviations-3.md (D6,
prereg-r6-d3) and r6-deviations-4.md (D7, prereg-r6-d4). Lens quantities come from
lens/geometry.py (docs/theory.md).

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
# inputs: D7 (3) row order                                                      #
# --------------------------------------------------------------------------- #


def episode_order_ok(env_seed, episode_in_seed, env_seeds, episodes_per_seed):
    """D7 (3): "Rows: the observations in stored order: environment seed, then
    episode within seed, then step. [...] Before using a file, the stage checks that
    its `env_seed` and `episode_in_seed` arrays are in this order, and stops if they
    are not." Checked against the protocol of D6 (e) (environment seeds 0-4, 10
    consecutive episodes each). episode_in_seed=None checks env_seed alone (the D4
    files store no episode index)."""
    want_s = np.repeat(np.asarray(env_seeds), episodes_per_seed)
    if not np.array_equal(np.asarray(env_seed), want_s):
        return False
    if episode_in_seed is None:
        return True
    want_e = np.tile(np.arange(episodes_per_seed), len(env_seeds))
    return bool(np.array_equal(np.asarray(episode_in_seed), want_e))


# --------------------------------------------------------------------------- #
# r6.md Definitions                                                             #
# --------------------------------------------------------------------------- #


def canonical_sign(v):
    """D7 (7): "Eigenvectors of C (d₁, d₂) and singular vectors of A (u_min and the
    principal directions) are defined up to sign. Each is signed so that its
    largest-magnitude component is positive. A tie in magnitude is broken by the
    lowest index." (np.argmax returns the lowest index among equal maxima.)"""
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
    (n, k) in layer-input coordinates. D6 (g): "r6.md has no kept-dimension step for
    d₁: d₁ is an eigenvector of the full C [...] d₂ is computed the same way. If d₂'s
    eigenvalue is below the kept-dimension threshold (1e-10 times the largest), d₂ is
    reported as degenerate for that checkpoint and its (g) quantities are not
    computed." """
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
    """r6.md: "Inside: w(z*, μ)² is at most the 95% quantile of χ² with m degrees of
    freedom." Also reported: the part of z* − μ outside the kept eigenvectors, which
    C⁺ ignores."""
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
    other row. D7 (3): ""Other" means another row index. An exact duplicate at another
    index counts, at distance 0." """
    from scipy.spatial.distance import cdist
    n = len(Y)
    out = np.empty(n)
    for i in range(0, n, chunk):
        Dm = cdist(Y[i:i + chunk], Y)  # direct differences, not the |a|^2 + |b|^2 - 2ab form
        Dm[np.arange(len(Dm)), np.arange(i, i + len(Dm))] = np.inf
        out[i:i + chunk] = Dm.min(axis=1)
    return out


def populated(S, X, z_star, n_sub=5000, rng_seed=0, percentile=95.0):
    """r6.md: "Populated: draw a subsample of 5,000 observations with
    numpy.random.default_rng(0).choice(n, 5000, replace=False), where n is the task's
    number of observations. For each subsample observation, compute the whitened
    distance to its nearest *other* subsample observation. The whitened distance from
    z* to its nearest subsample observation must be at most the 95th percentile of
    those distances."
    D7 (3): rows of X in stored order (environment seed, episode, step); "z*'s nearest
    neighbour is taken from the same 5,000 subsample observations, as r6.md says, not
    from all n observations"; "Whitened distances use C⁺ from all n observations";
    "The 95th percentile is numpy's default (linear) method."
    Returns (results, subsample indices, nearest-other distances)."""
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
                populated_nn_median=float(np.median(nn)), populated_n=int(n_sub)), idx, nn


def sharp(S, X, L, minimum=5.0):
    """r6.md: "Sharp: D / r_eff(d₁) ≥ 5." r_eff(d₁) = r*(d₁)·√(1 + κ) along d₁ through
    z* (convention 6; geometry.line)."""
    D = half_width_D(S, X)
    ln = geo.line(L, S.d1)
    ratio = D / ln["r_eff"]
    return dict(D=D, r_star_d1=float(ln["r_star"]), r_eff_d1=float(ln["r_eff"]),
                sharp_ratio=float(ratio), sharp=bool(ratio >= minimum))


def rho_fractions(L, X, levels=(1, 10, 100, 1000)):
    """r6.md, reported regardless: "the distribution of ρ_eff over the observations
    (fractions with ρ_eff ≤ 1, 10, 100 and 1,000), the same fractions for ρ when the
    lens is non-degenerate"."""
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
    """r6.md: "A task's lens is "sharp and in the data" if all three hold." Returns
    the three quantities, the conjunction, the subsample indices and the
    nearest-other distances."""
    out = {}
    out.update(inside(S, L.z_star, float(cfg_c["inside_quantile"])))
    pop, idx, nn = populated(S, X, L.z_star, int(cfg_c["populated"]["n_sub"]),
                             int(cfg_c["populated"]["rng_seed"]),
                             float(cfg_c["populated"]["percentile"]))
    out.update(pop)
    out.update(sharp(S, X, L, float(cfg_c["sharp_min"])))
    out["criterion"] = bool(out["inside"] and out["populated"] and out["sharp"])
    return out, idx, nn


# --------------------------------------------------------------------------- #
# D6 (b), D7 (1), (2): identification and the tie rule                          #
# --------------------------------------------------------------------------- #


def condition_ii(err, candidate="symlog", e_over_e0_max=0.1):
    """D6 (b) (ii): "on the new planner data, symlog gives the lowest e among the
    three D3 candidates, with e/e₀ ≤ 0.1 (D3's computation)."
    D7 (2): "e/e₀ ≤ 0.1 uses each candidate's own random-pairing baseline: e₀ is
    computed in that candidate's coordinates, with D3's indices
    j = numpy.random.default_rng(0).integers(0, n, size=n)."
    err: {candidate: dict(e, e0)} from collect.consistency_errors, whose e0 is each
    candidate's own. "Lowest": e_symlog ≤ every other e (as collect.decide)."""
    e = err[candidate]["e"]
    lowest = all(e <= err[c]["e"] for c in err)
    ratio = e / err[candidate]["e0"]
    return dict(lowest=bool(lowest), e_over_e0=float(ratio), threshold=float(e_over_e0_max),
                holds=bool(lowest and ratio <= e_over_e0_max))


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


def tie_rule(err, candidate="symlog", tie_ratio=1.1):
    """D7 (1): "Suppose symlog gives the lowest e among the three D3 candidates. A
    non-symlog reading c is within 10% of symlog if e_c / e_symlog ≤ 1.1. If at least
    one reading is within 10%, cartpole-swingup seed 1 is ambiguous."
    Returns every ratio (all are scanned by D7 (8)), the readings within 10%, and
    whether seed 1 is ambiguous (only when symlog gives the lowest e)."""
    e_s = err[candidate]["e"]
    lowest = all(e_s <= err[c]["e"] for c in err)
    ratios = {c: float(err[c]["e"] / e_s) for c in err if c != candidate}
    within = sorted(c for c, r in ratios.items() if r <= tie_ratio)
    return dict(symlog_lowest=bool(lowest), ratios=ratios, threshold=float(tie_ratio),
                within=within if lowest else [], ambiguous=bool(lowest and within))


def cartpole_seed1(identified, tie, criterion_by_reading, primary="symlog"):
    """D7 (1): "In that case it counts as passing for G1 only if the criterion holds
    under symlog and under every reading within 10%. If both non-symlog readings are
    within 10%, the criterion must hold under all three [...]. An ambiguous seed 1
    remains the counting checkpoint for cartpole-swingup: it does not hand the vote to
    seed 2. Seed 2 votes only if seed 1 is unidentified under D6 (b)."
    criterion_by_reading: {reading: bool} for cartpole-swingup seed 1 (D7 (5))."""
    if not identified:
        return dict(identified=False, ambiguous=False, required_readings=[], criterion=None)
    required = [primary] + [r for r in tie["within"] if r != primary]
    return dict(identified=True, ambiguous=bool(tie["ambiguous"]), required_readings=required,
                criterion=bool(all(criterion_by_reading[r] for r in required)))


def counting_checkpoints(tasks, criterion_primary, cartpole, flags,
                         cartpole_task="cartpole-swingup", counting_seed=1, fallback_seed=2):
    """The checkpoint that counts toward G1 for each task.
    r6.md, Models: "Seed 1 is the primary checkpoint for each task. [...] only seed 1
    counts toward the gate."
    D6 (a): "If this happens [planner return below 0.5 of published] to the checkpoint
    that counts toward G1 for a task (seed 1, or seed 2 for cartpole-swingup if seed 1
    ends up unidentified under (b)), that task leaves G1, and G1 then needs 2 of the
    remaining non-dog tasks."
    D7 (1): an identified seed 1 of cartpole-swingup votes with cartpole_seed1()'s
    result, ambiguous or not.
    criterion_primary: {(task, seed): bool} in the primary reading; flags:
    {(task, seed): below 0.5 x published}."""
    rows = []
    for t in tasks:
        if t == cartpole_task and cartpole["identified"]:
            seed, holds = counting_seed, cartpole["criterion"]
            why = "seed 1, identified under D6 (b)"
            if cartpole["ambiguous"]:
                why += "; ambiguous (D7 (1)): criterion required under " + ", ".join(cartpole["required_readings"])
        elif t == cartpole_task:
            seed, holds = fallback_seed, criterion_primary[(t, fallback_seed)]
            why = "seed 1 unidentified under D6 (b): seed 2"
        else:
            seed, holds, why = counting_seed, criterion_primary[(t, counting_seed)], "seed 1"
        flagged = bool(flags.get((t, seed), False))
        rows.append(dict(task=t, counting_seed=seed, why=why, criterion=bool(holds),
                         flagged_below_half=flagged, counts=not flagged))
    return rows


def g1_vote(rows, min_pass=2):
    """r6.md, Gate G1: "G1 passes if the criterion holds for at least 2 of the 4 tasks
    other than dog-run." With D6 (a): a flagged task leaves G1, which still needs 2."""
    n_counting = sum(r["counts"] for r in rows)
    n_pass = sum(r["counts"] and r["criterion"] for r in rows)
    return dict(rows=rows, n_counting=int(n_counting), n_pass=int(n_pass), min_pass=int(min_pass),
                g1=bool(n_pass >= min_pass))


# --------------------------------------------------------------------------- #
# D7 (8): borderline scan                                                       #
# --------------------------------------------------------------------------- #


def borderline_items(crit_rows, cii=None, tie=None, gate_name="cartpole-swingup-seed1"):
    """D7 (8): "The statistics are: Inside: w(z*, μ)² against the χ²_m 95% quantile;
    Populated: the whitened distance from z* against the 95th percentile; Sharp:
    D / r_eff(d₁) against 5; D6 (b) condition (ii), for cartpole-swingup seed 1 on the
    planner data: symlog's e/e₀ against 0.1; (1)'s ratio e_c / e_symlog against 1.1,
    for cartpole-swingup seed 1 on the planner data, for each non-symlog reading c."
    crit_rows: one per checkpoint and reading, with task, seed, reading and the
    criterion quantities."""
    items = []
    for r in crit_rows:
        where = f"{r['task']}-seed{r['seed']} [{r['reading']}]"
        items += [dict(where=where, statistic="Inside w2 vs chi2_m 95% quantile",
                       value=r["inside_w2"], threshold=r["inside_chi2_q"]),
                  dict(where=where, statistic="Populated distance vs 95th percentile",
                       value=r["populated_dist"], threshold=r["populated_threshold"]),
                  dict(where=where, statistic="Sharp D / r_eff(d1) vs 5",
                       value=r["sharp_ratio"], threshold=r["sharp_min"])]
    if cii is not None:
        items.append(dict(where=f"{gate_name} [planner data]", statistic="condition (ii) e/e0 vs 0.1",
                          value=cii["e_over_e0"], threshold=cii["threshold"]))
    if tie is not None:
        for c, ratio in sorted(tie["ratios"].items()):
            items.append(dict(where=f"{gate_name} [planner data]",
                              statistic=f"D7 (1) ratio e_{c} / e_symlog vs 1.1",
                              value=ratio, threshold=tie["threshold"]))
    return items


def borderline_scan(items, band=0.01):
    """D7 (8): "A statistic is borderline if it lies within 1% of its threshold:
    |s − τ| ≤ 0.01·|τ|." Returns the items with their relative distance and flag."""
    out = []
    for it in items:
        s, tau = float(it["value"]), float(it["threshold"])
        gap = abs(s - tau)
        rel = gap / abs(tau) if tau != 0 else (0.0 if gap == 0 else np.inf)
        out.append(dict(it, rel_distance=float(rel), borderline=bool(gap <= band * abs(tau))))
    return out


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
# Corollary 1 checks: D6 (g) lines through z*, D7 (4) lines through the data    #
# --------------------------------------------------------------------------- #


def ln_hat(E, b, eps, X):
    """Layer-1 LayerNorm output before γ and β (theory.md's ĥ) at inputs X (..., k)."""
    h = np.asarray(X, np.float64) @ np.asarray(E, np.float64).T + b
    r = h - h.mean(-1, keepdims=True)
    return r / np.sqrt((r ** 2).mean(-1, keepdims=True) + eps)


def _max_rel_dev(hh, cf):
    return float(np.max(np.linalg.norm(hh - cf, axis=-1) / np.linalg.norm(cf, axis=-1)))


def corollary1(L, d, t):
    """D6 (g): ĥ(t) = √H (cos θ ĉ + sin θ q̂) / √(1 + κ cos² θ), θ = arctan(t / r*(d)),
    q̂ = A d / ‖A d‖, along x(t) = z* + t d (non-degenerate lens)."""
    q = L.A @ d
    qh = q / np.linalg.norm(q)
    th = np.arctan(np.asarray(t, np.float64) / geo.line(L, d)["r_star"])[:, None]
    return np.sqrt(L.H) * (np.cos(th) * L.c_hat + np.sin(th) * qh) / np.sqrt(1.0 + L.kappa * np.cos(th) ** 2)


def theta_grid(L, d, X, n_grid=2001, t_range=3.0):
    """D6 (g) Lines: "x(t) = z* + t·d, with 2,001 points uniform in θ over
    |θ| ≤ arctan(T / r*(d)), where T = max(max_i |⟨x_i − z*, d⟩|, 3·r_eff(d)) over the
    data states x_i"."""
    ln = geo.line(L, d)
    T = max(float(np.max(np.abs((np.asarray(X, np.float64) - L.z_star) @ d))), t_range * ln["r_eff"])
    th = np.linspace(-np.arctan(T / ln["r_star"]), np.arctan(T / ln["r_star"]), n_grid)
    return ln["r_star"] * np.tan(th), dict(T=T, r_star=ln["r_star"], r_eff=ln["r_eff"])


def corollary1_g_line(L, E, b, d, X, n_grid=2001, t_range=3.0):
    """D6 (g): "the layer-1 LayerNorm output before its affine parameters γ and β
    (theory.md's ĥ) along each line matches the closed form above to 1e-10 relative
    (float64); the maximum deviation is reported." Returns the grid t, its info and
    the maximum relative deviation."""
    d = np.asarray(d, np.float64)
    t, info = theta_grid(L, d, X, n_grid, t_range)
    dev = _max_rel_dev(ln_hat(E, b, L.eps, L.z_star + t[:, None] * d), corollary1(L, d, t))
    return t, info, dev


def corollary1_data_line(L, E, b, x0, d, X, n_grid=2001, t_range=3.0):
    """D7 (4): "Each line x₀ + s·d uses its own closest approach s*, c⊥,ℓ, κ_ℓ and r*_ℓ
    (theory.md, Corollary 1), not the lens's. The grid follows D6 (g)'s rule, about
    the line's own closest approach x₀ + s*·d: 2,001 points uniform in θ over
    |θ| ≤ arctan(T / r*_ℓ), with T = max(max_i |⟨x_i − (x₀ + s*·d), d⟩|, 3·r_eff,ℓ).
    At each grid point, the deviation is ‖ĥ − ĥ_Cor1‖₂ / ‖ĥ_Cor1‖₂."
    theory.md, Corollary 1: θ(s) = arctan((s − s*) / r*_ℓ),
    ĥ(s) = √H (cos θ ĉ_ℓ + sin θ q̂) / √(1 + κ_ℓ cos² θ)."""
    d = np.asarray(d, np.float64)
    x0 = np.asarray(x0, np.float64)
    ln = geo.line(L, d, x0)
    if ln["norm_c_perp_l"] == 0.0:
        raise ValueError("the line passes through the degenerate set c_perp_l = 0.")
    xc = x0 + ln["s_star"] * d
    rs = ln["r_star"]
    T = max(float(np.max(np.abs((np.asarray(X, np.float64) - xc) @ d))), t_range * ln["r_eff"])
    th = np.linspace(-np.arctan(T / rs), np.arctan(T / rs), n_grid)
    tau = rs * np.tan(th)
    q = L.A @ d
    qh = q / np.linalg.norm(q)
    ch = ln["c_perp_l"] / ln["norm_c_perp_l"]
    c = np.cos(th)[:, None]
    cf = np.sqrt(L.H) * (c * ch + np.sin(th)[:, None] * qh) / np.sqrt(1.0 + ln["kappa_l"] * c ** 2)
    dev = _max_rel_dev(ln_hat(E, b, L.eps, xc + tau[:, None] * d), cf)
    return dict(s_star=float(ln["s_star"]), r_star_l=float(rs), r_eff_l=float(ln["r_eff"]),
                kappa_l=float(ln["kappa_l"]), T=float(T), n_grid=int(n_grid), max_rel_dev=dev)


def data_line_rows(n, n_lines=10, rng_seed=0):
    """D7 (4): "10 lines along d₁, one through each of 10 data states. The states are
    the rows numpy.random.default_rng(0).choice(n, 10, replace=False), in the row order
    of (3)." """
    return np.random.default_rng(rng_seed).choice(n, n_lines, replace=False)


# --------------------------------------------------------------------------- #
# D6 (g): end-to-end susceptibility                                             #
# --------------------------------------------------------------------------- #


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


def susceptibility(L, d, t, info, outputs, t_range=3.0):
    """D6 (g) for one direction d (unit, layer-input coordinates), on the grid t of
    corollary1_g_line (whose check has already passed).

    D6 (g), Reported per checkpoint, direction and output: "‖g′(0)‖; S = ‖g′(0)‖ /
    r*(d), with the r_eff(d) version ‖g′(0)‖ / r_eff(d) beside it [...]; the peak ‖J(t)‖
    over the grid points with |t| ≤ 3·r_eff(d), and the t where it occurs; the median
    over data states of ‖J(x_i) d‖ with a = 0, and a secondary version using each
    state's recorded executed action; the ratio of the peak to that median (a = 0)."
    with g′(θ) = J(t) · (r*(d)² + t²) / r*(d).

    outputs: {name: (f_line, {key: (f_data, rows, direction)})}. f_line maps a layer
    input (k,) to the output, with the action fixed at a = 0 for the dynamics. For
    the a = 0 version the rows are the states x_i (all 25,050, D7 (6)) and the
    direction d; for the recorded-action version the rows are [x_i, a_i] (the 25,000
    states with an executed action, D7 (6)) and the direction [d, 0].
    """
    d = np.asarray(d, np.float64)
    line = L.z_star + t[:, None] * d
    rs = info["r_star"]
    near = np.abs(t) <= t_range * info["r_eff"]
    res = {}
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
        res[name] = o
    return res
