"""The two TD-MPC2 checkpoint layouts at HF revision 8fb2a82 (deviation D5).

public       NormedLinear modules, as built by tdmpc2 e9f59321 common/layers.py:
             <i>.weight, <i>.bias, <i>.ln.weight, <i>.ln.bias per hidden layer;
             forward Linear -> (dropout) -> LayerNorm -> act (lines 107-111); every
             layer but the last acts with Mish; the last acts with the network's
             final activation (SimNorm), or is a plain Linear if there is none
             (mlp(), lines 128-133).
prerelease   flat Sequential: Linear <i>, LayerNorm-shaped <i+1>, parameter-free
             activation <i+2>; encoder position 0 parameter-free (D1-D3).

detect_layout() decides from the keys and stops on anything else. Networks are
evaluated in float64 numpy.
"""

import sys

import numpy as np

PUBLIC, PRERELEASE = "public", "prerelease"


def detect_layout(sd, where=""):
    """'public' or 'prerelease' from the encoder's keys; stop on anything else."""
    def has(k):
        return k in sd

    enc0 = [k for k in sd if k.startswith("_encoder.state.0.")]
    if (has("_encoder.state.0.weight") and has("_encoder.state.0.ln.weight")
            and np.ndim(sd["_encoder.state.0.weight"]) == 2):
        return PUBLIC
    if (not enc0 and has("_encoder.state.1.weight") and has("_encoder.state.2.weight")
            and np.ndim(sd["_encoder.state.1.weight"]) == 2
            and np.ndim(sd["_encoder.state.2.weight"]) == 1):
        return PRERELEASE
    enc = sorted(k for k in sd if k.startswith("_encoder."))
    sys.exit(f"{where}: encoder layout is neither public nor pre-release; keys {enc[:12]}")


def first_layer_keys(layout, cfg_layouts):
    """(linear prefix, norm prefix) of the encoder's first layer for a layout."""
    lay = cfg_layouts[layout]
    return lay["linear"], lay["norm"]


def check_first_layer(sd, layout, cfg_layouts, where=""):
    """Stop unless the encoder's first layer has exactly the expected keys and shapes."""
    lin, nrm = first_layer_keys(layout, cfg_layouts)
    errs = []

    def under(prefix, exclude=()):
        return sorted(k[len(prefix) + 1:] for k in sd if k.startswith(prefix + ".")
                      and not any(k.startswith(prefix + "." + e + ".") for e in exclude))

    if layout == PUBLIC:
        if under(lin, exclude=("ln",)) != ["bias", "weight"]:
            errs.append(f"{lin}: keys {under(lin)}")
        if under(nrm) != ["bias", "weight"]:
            errs.append(f"{nrm}: keys {under(nrm)}")
    else:
        if under(lin) != ["bias", "weight"]:
            errs.append(f"{lin}: keys {under(lin)}, expected bias, weight")
        if under(nrm) != ["bias", "weight"]:
            errs.append(f"{nrm}: keys {under(nrm)}, expected bias, weight")
        for p in cfg_layouts[layout].get("param_free", []):
            if under(p):
                errs.append(f"{p}: expected no parameters or buffers, found {under(p)}")
    if not errs:
        W, b = sd[f"{lin}.weight"], sd[f"{lin}.bias"]
        g, be = sd[f"{nrm}.weight"], sd[f"{nrm}.bias"]
        H = W.shape[0]
        if np.ndim(W) != 2 or tuple(b.shape) != (H,):
            errs.append(f"{lin}: weight {tuple(W.shape)}, bias {tuple(b.shape)}")
        if tuple(g.shape) != (H,) or tuple(be.shape) != (H,):
            errs.append(f"{nrm}: weight {tuple(g.shape)}, bias {tuple(be.shape)}, expected ({H},)")
    if errs:
        sys.exit(f"{where}: checkpoint does not have the {layout} layout:\n  " + "\n  ".join(errs))
    return lin, nrm


# --------------------------------------------------------------------------- #
# activations and input candidates (D3)                                         #
# --------------------------------------------------------------------------- #


def mish(x):
    return x * np.tanh(np.logaddexp(0.0, x))


def simnorm(x, dim):
    s = x.shape
    x = x.reshape(*s[:-1], -1, dim)
    x = np.exp(x - x.max(-1, keepdims=True))
    return (x / x.sum(-1, keepdims=True)).reshape(s)


def layernorm(x, g, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def input_candidates(eps):
    """D3 candidates for what the encoder's first Linear receives."""
    def ln_noaffine(o):
        m = o.mean(-1, keepdims=True)
        return (o - m) / np.sqrt(((o - m) ** 2).mean(-1, keepdims=True) + eps)
    return {"identity": lambda o: o,
            "symlog": lambda o: np.sign(o) * np.log1p(np.abs(o)),
            "layernorm": ln_noaffine}


# --------------------------------------------------------------------------- #
# public layout                                                                 #
# --------------------------------------------------------------------------- #


def build_public(sd, prefix, final, eps, simnorm_dim, module_names=None, input_fn=None):
    """mlp() of tdmpc2 e9f59321 from its state_dict: children 0..n-1, each a
    NormedLinear (weight, bias, ln.weight, ln.bias) except possibly a plain Linear
    last (only when final is None). Dropout, if present, is the identity in
    evaluation and has no parameters."""
    keys = [k for k in sd if k.startswith(prefix + ".")]
    idx = sorted({int(k[len(prefix) + 1:].split(".")[0]) for k in keys})
    errs = []
    if idx != list(range(len(idx))) or not idx:
        errs.append(f"{prefix}: children {idx}, expected 0..n-1")
    layers = []
    for i in idx:
        sub = sorted(k[len(f"{prefix}.{i}") + 1:] for k in keys if k.startswith(f"{prefix}.{i}."))
        last = i == idx[-1]
        if sub == ["bias", "ln.bias", "ln.weight", "weight"]:
            layers.append(("normed", i))
        elif sub == ["bias", "weight"] and last and final is None:
            layers.append(("linear", i))
        else:
            errs.append(f"{prefix}.{i}: keys {sub}")
    if layers and final is not None and layers[-1][0] != "normed":
        errs.append(f"{prefix}: last layer must be a NormedLinear (final activation {final})")
    if layers and final is None and layers[-1][0] != "linear":
        errs.append(f"{prefix}: last layer must be a plain Linear (no final activation)")
    if module_names is not None:
        allowed = {f"{i}" for i in idx} | {f"{i}.{s}" for i in idx for s in ("ln", "act", "dropout")}
        named = {m[len(prefix) + 1:] for m in module_names if m.startswith(prefix + ".")}
        extra = sorted(named - allowed)
        if extra:
            errs.append(f"{prefix}: unexpected modules {extra}")
    if errs:
        sys.exit("layout check failed (public):\n  " + "\n  ".join(errs))

    P = {i: dict(W=sd[f"{prefix}.{i}.weight"], b=sd[f"{prefix}.{i}.bias"],
                 g=sd.get(f"{prefix}.{i}.ln.weight"), be=sd.get(f"{prefix}.{i}.ln.bias"))
         for i in idx}

    def f(x):
        x = np.asarray(x, np.float64)
        if input_fn is not None:
            x = input_fn(x)
        for kind, i in layers:
            p = P[i]
            x = x @ p["W"].T + p["b"]
            if kind == "normed":
                x = layernorm(x, p["g"], p["be"], eps)
                x = simnorm(x, simnorm_dim) if (i == idx[-1] and final == "simnorm") else mish(x)
        return x

    f.ops = [(kind, str(i), tuple(P[i]["W"].shape)) for kind, i in layers]
    f.implied_free = []
    return f


# --------------------------------------------------------------------------- #
# pre-release layout                                                            #
# --------------------------------------------------------------------------- #


def _path_key(p):
    return tuple(int(x) for x in p.split("."))


def build_prerelease(sd, prefix, final, eps, simnorm_dim, module_names=None, pos0=None):
    """Sequential network from the parametrised modules under `prefix`.

    A module with a 2-D weight is a Linear; a 1-D weight with a bias is a
    LayerNorm (eps). After each LayerNorm: Mish, or `final` ('simnorm') after the
    network's last module. Index gaps must be exactly the parameter-free
    positions this rule implies (plus position 0 for the encoder, `pos0`); if the
    state_dict carries module names they must agree too. Anything else stops.
    """
    keys = [k for k in sd if k.startswith(prefix + ".")]
    paths = sorted({k[len(prefix) + 1:].rsplit(".", 1)[0] for k in keys}, key=_path_key)
    ops, errs = [], []
    for p in paths:
        w, b = sd.get(f"{prefix}.{p}.weight"), sd.get(f"{prefix}.{p}.bias")
        extra = sorted(k for k in keys if k.startswith(f"{prefix}.{p}.")
                       and k.rsplit(".", 1)[1] not in ("weight", "bias"))
        if w is None or b is None or extra:
            errs.append(f"{prefix}.{p}: keys {sorted(k for k in keys if k.startswith(prefix + '.' + p + '.'))}")
            continue
        ops.append(("linear" if np.ndim(w) == 2 else "ln", p, w, b))
    implied = []
    for kind, p, _, _ in ops:
        if kind == "ln":
            parent, idx = (p.rsplit(".", 1) if "." in p else ("", p))
            implied.append(f"{parent}.{int(idx) + 1}" if parent else str(int(idx) + 1))
    if pos0 is not None:
        implied.append("0")
    present = {p for _, p, _, _ in ops}
    for q in implied:
        if q in present:
            errs.append(f"{prefix}.{q}: rule implies a parameter-free module, found parameters")
    for kind, p, _, _ in ops:
        parent, idx = (p.rsplit(".", 1) if "." in p else ("", p))
        for j in range(int(idx)):
            q = f"{parent}.{j}" if parent else str(j)
            if q not in present and q not in implied and not any(
                    x.startswith(q + ".") for x in present):
                errs.append(f"{prefix}.{q}: unexplained parameter-free position")
    if module_names is not None:
        named = {m[len(prefix) + 1:] for m in module_names if m.startswith(prefix + ".")}
        leaves = {m for m in named if not any(o.startswith(m + ".") for o in named)}
        free = leaves - present
        if free != set(implied):
            errs.append(f"{prefix}: parameter-free modules {sorted(free, key=_path_key)} "
                        f"!= implied {sorted(set(implied), key=_path_key)}")
    if errs:
        sys.exit("layout check failed (pre-release):\n  " + "\n  ".join(errs))

    def f(x):
        x = np.asarray(x, np.float64)
        if pos0 is not None:
            x = pos0(x)
        for i, (kind, p, w, b) in enumerate(ops):
            if kind == "linear":
                x = x @ w.T + b
            else:
                x = layernorm(x, w, b, eps)
                x = simnorm(x, simnorm_dim) if (i == len(ops) - 1 and final == "simnorm") else mish(x)
        return x

    f.ops = [(k, p, tuple(np.shape(w))) for k, p, w, _ in ops]
    f.implied_free = sorted(set(implied), key=_path_key)
    return f


def build_networks(sd, layout, eps, simnorm_dim, module_names=None, input_fn=None):
    """Encoder (with input_fn at its input: position 0 for the pre-release layout,
    before the first Linear for the public one), dynamics and policy."""
    if layout == PUBLIC:
        b = build_public
        enc = b(sd, "_encoder.state", "simnorm", eps, simnorm_dim, module_names, input_fn=input_fn)
    elif layout == PRERELEASE:
        b = build_prerelease
        enc = b(sd, "_encoder.state", "simnorm", eps, simnorm_dim, module_names,
                pos0=input_fn if input_fn is not None else (lambda o: o))
    else:
        sys.exit(f"unknown layout {layout}")
    dyn = b(sd, "_dynamics", "simnorm", eps, simnorm_dim, module_names)
    pi = b(sd, "_pi", None, eps, simnorm_dim, module_names)
    return enc, dyn, pi
