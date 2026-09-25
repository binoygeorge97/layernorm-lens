"""D6 (a), (b), (e): the logic of the planner collector that needs neither torch nor
tdmpc2, so that it can be tested without checkpoints. numpy float64 throughout.

- PRERELEASE_REMAP / remap_prerelease(): D6 (b)'s key remap of the two pre-release
  checkpoints to the public layout (40 keys renamed; the 20 Q-ensemble keys pass
  through to tdmpc2's own api_model_conversion).
- public_shapes(): the public WorldModel's parameter names and shapes for a 5M
  single-task model, from tdmpc2 e9f59321 common/world_model.py and layers.py.
- planner_input(): what the planner receives: the raw observation (public layout)
  or symlog of it (pre-release), computed with layouts.input_candidates in float64
  and cast to float32.
- random_states(), relative_errors(), gate_rule(), control_outcomes(), halts():
  the encoder agreement gate and its controls, and the literal halting rule.
- d1(), lens_distance(): d1 and ||x - z*|| / r_eff(d1) for the worst states.
- lens_geometry(): lens/geometry.py loaded by file path (lens/__init__ imports
  core, which imports JAX; tdmpc2's environment has no JAX).
"""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import layouts  # noqa: E402

# --------------------------------------------------------------------------- #
# D6 (b): key remap                                                             #
# --------------------------------------------------------------------------- #

# (row of D6's table, as printed there; [(pre-release prefix, public prefix)])
PRERELEASE_REMAP = [
    ("| `_encoder.state.1`, `.2` | `_encoder.state.0`, `.0.ln` |",
     [("_encoder.state.1", "_encoder.state.0"), ("_encoder.state.2", "_encoder.state.0.ln")]),
    ("| `_encoder.state.4`, `.5` | `_encoder.state.1`, `.1.ln` |",
     [("_encoder.state.4", "_encoder.state.1"), ("_encoder.state.5", "_encoder.state.1.ln")]),
    ("| `_dynamics.0.0`, `.0.1` | `_dynamics.0`, `.0.ln` |",
     [("_dynamics.0.0", "_dynamics.0"), ("_dynamics.0.1", "_dynamics.0.ln")]),
    ("| `_dynamics.0.3`, `.0.4` | `_dynamics.1`, `.1.ln` |",
     [("_dynamics.0.3", "_dynamics.1"), ("_dynamics.0.4", "_dynamics.1.ln")]),
    ("| `_dynamics.0.6`, `_dynamics.1` | `_dynamics.2`, `.2.ln` |",
     [("_dynamics.0.6", "_dynamics.2"), ("_dynamics.1", "_dynamics.2.ln")]),
    ("| `_reward.0`, `.1` | `_reward.0`, `.0.ln` |",
     [("_reward.0", "_reward.0"), ("_reward.1", "_reward.0.ln")]),
    ("| `_reward.3`, `.4` | `_reward.1`, `.1.ln` |",
     [("_reward.3", "_reward.1"), ("_reward.4", "_reward.1.ln")]),
    ("| `_reward.6` | `_reward.2` (plain Linear, 101 bins) |",
     [("_reward.6", "_reward.2")]),
    ("| `_pi.0`, `.1` | `_pi.0`, `.0.ln` |",
     [("_pi.0", "_pi.0"), ("_pi.1", "_pi.0.ln")]),
    ("| `_pi.3`, `.4` | `_pi.1`, `.1.ln` |",
     [("_pi.3", "_pi.1"), ("_pi.4", "_pi.1.ln")]),
    ("| `_pi.6` | `_pi.2` (plain Linear, 2 × action dim) |",
     [("_pi.6", "_pi.2")]),
]
PREFIX_MAP = [p for _, pairs in PRERELEASE_REMAP for p in pairs]
Q_PREFIXES = ("_Qs.params.", "_target_Qs.params.")


class RemapError(ValueError):
    pass


def remap_keys(keys):
    """{pre-release key: public key} for every key of a pre-release state_dict.

    The 40 keys under D6's prefixes are renamed (.weight and .bias of each); the 20
    Q-ensemble keys are kept for api_model_conversion. Any other key, a missing key
    or a collision stops.
    """
    keys = list(keys)
    out, errs = {}, []
    wanted = {f"{old}.{s}": f"{new}.{s}" for old, new in PREFIX_MAP for s in ("weight", "bias")}
    for k in keys:
        if k in wanted:
            out[k] = wanted[k]
        elif k.startswith(Q_PREFIXES) and k.rsplit(".", 1)[1].isdigit():
            out[k] = k
        else:
            errs.append(f"unexpected key {k}")
    errs += [f"missing key {k}" for k in sorted(set(wanted) - set(keys))]
    q = sorted(k for k in keys if k.startswith(Q_PREFIXES))
    want_q = sorted(f"{p}{i}" for p in Q_PREFIXES for i in range(10))
    if q != want_q:
        errs.append(f"Q-ensemble keys {q} != {want_q}")
    if len(set(out.values())) != len(out):
        errs.append("remap is not injective")
    if errs:
        raise RemapError("pre-release remap: " + "; ".join(errs))
    return out


def remap_prerelease(sd):
    """New dict with the remapped keys (values unchanged), and the key mapping."""
    m = remap_keys(sd.keys())
    return {m[k]: v for k, v in sd.items()}, m


def public_shapes(k, a, enc_dim=256, mlp_dim=512, latent=512, num_bins=101):
    """Parameter names and shapes of the public WorldModel's encoder, dynamics,
    reward and policy (tdmpc2 e9f59321: world_model.py lines 24-29, layers.enc and
    layers.mlp) for observation dimension k and action dimension a."""
    def normed(prefix, i, fan_in, fan_out):
        return {f"{prefix}.{i}.weight": (fan_out, fan_in), f"{prefix}.{i}.bias": (fan_out,),
                f"{prefix}.{i}.ln.weight": (fan_out,), f"{prefix}.{i}.ln.bias": (fan_out,)}

    def linear(prefix, i, fan_in, fan_out):
        return {f"{prefix}.{i}.weight": (fan_out, fan_in), f"{prefix}.{i}.bias": (fan_out,)}

    s = {}
    s.update(normed("_encoder.state", 0, k, enc_dim))
    s.update(normed("_encoder.state", 1, enc_dim, latent))
    for i, fin in enumerate([latent + a, mlp_dim, mlp_dim]):
        s.update(normed("_dynamics", i, fin, latent if i == 2 else mlp_dim))
    s.update(normed("_reward", 0, latent + a, mlp_dim))
    s.update(normed("_reward", 1, mlp_dim, mlp_dim))
    s.update(linear("_reward", 2, mlp_dim, num_bins))
    s.update(normed("_pi", 0, latent, mlp_dim))
    s.update(normed("_pi", 1, mlp_dim, mlp_dim))
    s.update(linear("_pi", 2, mlp_dim, 2 * a))
    return s


# --------------------------------------------------------------------------- #
# input and states                                                              #
# --------------------------------------------------------------------------- #


def input_fn(name, pos0_eps):
    """float64 map from the raw observation to what the first Linear receives."""
    return layouts.input_candidates(pos0_eps)[name]


def planner_input(obs, name, pos0_eps):
    """What the planner receives: input_fn in float64, cast to float32."""
    return np.asarray(input_fn(name, pos0_eps)(np.asarray(obs, np.float64)), np.float32)


def random_states(obs, n, seed, ddof):
    """D6 (b) set (i): default_rng(seed).standard_normal((n, k)), scaled by the
    per-dimension standard deviation (ddof) and shifted by the mean of `obs`, in raw
    observation coordinates."""
    X = np.asarray(obs, np.float64).reshape(-1, np.shape(obs)[-1])
    z = np.random.default_rng(seed).standard_normal((n, X.shape[1]))
    return z * X.std(axis=0, ddof=ddof) + X.mean(axis=0)


def relative_errors(z_planner, z_lens):
    """Per-state ||z_planner - z_lens||_2 / ||z_lens||_2, in float64."""
    zp = np.asarray(z_planner, np.float64)
    zl = np.asarray(z_lens, np.float64)
    return np.linalg.norm(zp - zl, axis=-1) / np.linalg.norm(zl, axis=-1)


# --------------------------------------------------------------------------- #
# gate rule, controls, halting                                                  #
# --------------------------------------------------------------------------- #


def summarise(err):
    err = np.asarray(err, np.float64)
    return dict(n=int(err.size), median=float(np.median(err)),
                p99=float(np.percentile(err, 99)), max=float(err.max()))


def gate_rule(err, median_max, max_max):
    """Pass iff median <= median_max and max <= max_max (D6 (b))."""
    s = summarise(err)
    s.update(median_ok=s["median"] <= median_max, max_ok=s["max"] <= max_max)
    s["pass"] = bool(s["median_ok"] and s["max_ok"])
    return s


def evaluate_sets(errs_by_set, g):
    """gate_rule on each state set; passes only if every set passes."""
    sets = {name: gate_rule(e, g["median_max"], g["max_max"]) for name, e in errs_by_set.items()}
    return dict(sets=sets, **{"pass": all(s["pass"] for s in sets.values())})


def control_outcomes(positive, negative, negative_min_max):
    """Whether each control behaves as D6 states.

    positive: evaluate_sets result; must pass on every set.
    negative: evaluate_sets result; must fail by a wide margin: the maximum relative
    error exceeds negative_min_max on every set (which also fails the rule).
    """
    pos_ok = bool(positive["pass"])
    neg_ok = all((not s["pass"]) and s["max"] > negative_min_max
                 for s in negative["sets"].values())
    return dict(positive_as_stated=pos_ok, negative_as_stated=neg_ok,
                as_stated=pos_ok and neg_ok)


def halts(controls, gate):
    """The literal halting rule (author, 25 Sep 2026).

    A control that does not behave as stated halts everything, public collection
    included. A failed gate halts only the pre-release runs (D6 (b)).
    controls: control_outcomes() or None if not run; gate: {name: evaluate_sets()}
    or None if not run.
    """
    reasons = []
    public_ok = prerelease_ok = True
    if controls is None or not controls["as_stated"]:
        public_ok = prerelease_ok = False
        if controls is None:
            reasons.append("controls not run")
        else:
            if not controls["positive_as_stated"]:
                reasons.append("positive control did not pass the rule")
            if not controls["negative_as_stated"]:
                reasons.append("negative control did not fail by the stated margin")
    if gate is None:
        prerelease_ok = False
        reasons.append("encoder agreement gate not run")
    else:
        failed = sorted(n for n, r in gate.items() if not r["pass"])
        if failed:
            prerelease_ok = False
            reasons.append(f"encoder agreement gate failed for {failed}")
    return dict(public_allowed=public_ok, prerelease_allowed=prerelease_ok, reasons=reasons)


# --------------------------------------------------------------------------- #
# lens distance of the worst states                                             #
# --------------------------------------------------------------------------- #


def lens_geometry():
    """lens/geometry.py as a module, without importing the lens package."""
    path = os.path.join(ROOT, "lens", "geometry.py")
    spec = importlib.util.spec_from_file_location("lens_geometry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def d1(X):
    """r6.md's d1: the unit eigenvector of C (numpy.cov, ddof = 1) with the largest
    eigenvalue, for observations X (n, k) in layer-input coordinates."""
    C = np.cov(np.asarray(X, np.float64), rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    return V[:, np.argmax(w)]


def lens_distance(X, E, b, eps, d):
    """||x - z*||_2 / r_eff(d) for each row x of X (layer-input coordinates), with z*
    and r_eff(d) from docs/theory.md via lens/geometry.py."""
    geo = lens_geometry()
    L = geo.lens(E, b, eps)
    r = geo.line(L, d)["r_eff"]
    dist = np.linalg.norm(np.asarray(X, np.float64) - L.z_star, axis=-1)
    return dist / r, dict(r_eff_d1=float(r), norm_z_star=float(np.linalg.norm(L.z_star)),
                          kappa=float(L.kappa), degenerate=bool(L.degenerate))


def worst_states(err, X, E, b, eps, d, n):
    """The n states with the largest relative error: index, error, distance to z*
    in units of r_eff(d)."""
    idx = np.argsort(np.asarray(err))[::-1][:n]
    dist, info = lens_distance(np.asarray(X)[idx], E, b, eps, d)
    return [dict(index=int(i), rel_error=float(err[i]), dist_over_r_eff_d1=float(q))
            for i, q in zip(idx, dist)], info


# --------------------------------------------------------------------------- #
# returns                                                                       #
# --------------------------------------------------------------------------- #


def return_summary(returns, published, flag_below):
    R = np.asarray(returns, np.float64)
    frac = float(R.mean() / published) if published else float("nan")
    return dict(episodes=int(R.size), return_mean=float(R.mean()),
                return_std=float(R.std(ddof=1)) if R.size > 1 else float("nan"),
                return_min=float(R.min()), return_max=float(R.max()), published=published,
                fraction=frac, flag_below_half=bool(np.isfinite(frac) and frac < flag_below))
