"""Initial-lens check: the lens of LN(E z + b) at PyTorch's default initialisation.

    python experiments/initial_lens/run.py --config experiments/initial_lens/config.yaml

Uses lens/geometry.py only. Reports, per (H, k, eps, init), the quantities of
docs/theory.md implementation convention 4:
  torch_default: median over random unit d of r*(d) = |c_perp|/|A d|, the principal
                 widths |c_perp|/s_i, |z*| and kappa; median r* is compared with the
                 estimate sqrt(1 - (k + 1)/H).
  zero_bias:     every draw must be degenerate (c_perp = 0); the eps-limited
                 principal widths sqrt(H eps)/s_i are reported instead of r* (= 0).
  flax_default:  Flax's Dense init (lecun_normal kernel, zero bias) at Flax's
                 LayerNorm eps only; degenerate like zero_bias, reported the same way.

Writes to out_dir: summary.csv, draws.npz (per-draw values), hist_H{H}_k{k}.png,
config.yaml and meta.json (git commit, versions).
"""

import argparse
import csv
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lens import geometry as geo  # noqa: E402

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def draw(seed, H, k, i, n_dir):
    """One draw: E, b under PyTorch's nn.Linear(k, H) default, and n_dir unit d."""
    rng = np.random.default_rng([seed, H, k, i])
    lim = 1.0 / np.sqrt(k)
    E = rng.uniform(-lim, lim, (H, k))
    b = rng.uniform(-lim, lim, H)
    d = rng.standard_normal((n_dir, k))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return E, b, d


def draw_flax(seed, H, k, i):
    """Flax Dense(H) on k inputs: kernel (k, H) from lecun_normal() (Flax's
    default_kernel_init is exactly jax.nn.initializers.lecun_normal()), zero bias.
    Flax computes y = x @ kernel, so E = kernel^T."""
    key = jax.random.PRNGKey(seed)
    for x in (H, k, i):
        key = jax.random.fold_in(key, x)
    kernel = jax.nn.initializers.lecun_normal()(key, (k, H), jnp.float64)
    return np.asarray(kernel).T, np.zeros(H)


def runs(cfg):
    """(eps, init) pairs: inits (a), (b) at every eps; (c) at Flax's own eps only."""
    pairs = [(e, init) for e in cfg["eps"] for init in cfg["inits"]]
    return pairs + [(e, "flax_default") for e in cfg["flax_default"]["eps"]]


def run_shape(cfg, H, k):
    """Per-draw values for every (eps, init) of one (H, k)."""
    n, n_dir, seed = cfg["n_draws"], cfg["n_directions"], cfg["seed"]
    out = {}
    for eps, init in runs(cfg):
        rec = dict(degenerate=np.zeros(n, bool), median_r_star=np.full(n, np.nan),
                   principal_widths=np.full((n, k), np.nan),
                   principal_widths_eff=np.full((n, k), np.nan),
                   norm_z_star=np.full(n, np.nan), kappa=np.full(n, np.nan))
        for i in range(n):
            E, b, d = draw(seed, H, k, i, n_dir)
            if init == "zero_bias":
                b = np.zeros(H)
            elif init == "flax_default":
                E, b = draw_flax(seed, H, k, i)
            L = geo.lens(E, b, eps)
            rec["degenerate"][i] = L.degenerate
            rec["median_r_star"][i] = np.median([geo.width(L, dj) for dj in d])
            rec["principal_widths"][i] = L.principal_widths
            rec["principal_widths_eff"][i] = L.principal_widths_eff
            rec["norm_z_star"][i] = np.linalg.norm(L.z_star)
            rec["kappa"][i] = L.kappa
        out[(eps, init)] = rec
    return out


def stats(x):
    x = np.asarray(x, float).ravel()
    q = np.quantile(x, QUANTILES)
    return dict(n=x.size, mean=x.mean(), std=x.std(ddof=1), min=x.min(),
                q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4], max=x.max())


def summary_rows(H, k, res):
    est = float(np.sqrt(1.0 - (k + 1) / H))
    rows = []
    for (eps, init), r in res.items():
        base = dict(H=H, k=k, eps=eps, init=init,
                    n_degenerate=int(r["degenerate"].sum()), n_draws=r["degenerate"].size)
        if init == "torch_default":
            items = [("median_r_star", r["median_r_star"]),
                     ("principal_width", r["principal_widths"]),
                     ("principal_width_min", r["principal_widths"].min(1)),
                     ("principal_width_max", r["principal_widths"].max(1)),
                     ("norm_z_star", r["norm_z_star"]),
                     ("kappa", r["kappa"])]
        else:
            items = [("eps_principal_width", r["principal_widths_eff"]),
                     ("eps_principal_width_min", r["principal_widths_eff"].min(1)),
                     ("eps_principal_width_max", r["principal_widths_eff"].max(1))]
        for name, x in items:
            row = dict(base, quantity=name, **stats(x))
            if name == "median_r_star":
                row["estimate"] = est
                row["q50_over_estimate"] = row["q50"] / est
            rows.append(row)
    return rows


def plot_shape(H, k, res, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False})
    e0 = cfg["eps"][0]
    tb = res[(e0, "torch_default")]  # r*, widths, z* are eps-free
    est = np.sqrt(1.0 - (k + 1) / H)
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2), facecolor="#fcfcfb")

    def hist(a, x, color, label=None, bins=40, logx=False):
        a.hist(np.asarray(x).ravel(), bins=bins, color=color, alpha=0.75 if label else 1.0,
               label=label, edgecolor="#fcfcfb", linewidth=0.6)
        if logx:
            a.set_xscale("log")
            a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.grid(axis="y", color=grid, linewidth=0.6)
        a.set_axisbelow(True)
        a.set_facecolor("#fcfcfb")

    a = ax[0, 0]
    hist(a, tb["median_r_star"], c1)
    a.axvline(est, color=ink, linestyle="--", linewidth=1.2)
    a.text(est, a.get_ylim()[1] * 0.95, f"  estimate {est:.3f}", color=ink, va="top")
    a.set_title("(b) median over d of r*(d)", color=ink, loc="left")
    a.set_xlabel("input units")

    hist(ax[0, 1], tb["principal_widths"], c1)
    ax[0, 1].set_title(f"(b) principal widths |c⊥|/sᵢ ({k} per draw)", color=ink, loc="left")
    ax[0, 1].set_xlabel("input units")

    hist(ax[0, 2], tb["norm_z_star"], c1)
    ax[0, 2].set_title("(b) |z*|", color=ink, loc="left")
    ax[0, 2].set_xlabel("input units")

    def logbins(key, init):  # one shared set of log bins for all eps series
        x = np.concatenate([res[(e, init)][key].ravel() for e in cfg["eps"]])
        return np.geomspace(x.min(), x.max(), 81)

    bk = logbins("kappa", "torch_default")
    xw = np.concatenate([res[r]["principal_widths_eff"].ravel() for r in res
                         if r[1] != "torch_default"])
    bw = np.geomspace(xw.min(), xw.max(), 81)
    for eps, c in zip(cfg["eps"], (c1, c2)):
        hist(ax[1, 0], res[(eps, "torch_default")]["kappa"], c, label=f"ε = {eps:g}",
             bins=bk, logx=True)
        hist(ax[1, 1], res[(eps, "zero_bias")]["principal_widths_eff"], c,
             label=f"zero bias, ε = {eps:g}", bins=bw, logx=True)
    for eps in cfg["flax_default"]["eps"]:
        hist(ax[1, 1], res[(eps, "flax_default")]["principal_widths_eff"], c3,
             label=f"(c) Flax, ε = {eps:g}", bins=bw, logx=True)
    ax[1, 0].set_title("(b) κ = Hε/|c⊥|²", color=ink, loc="left")
    ax[1, 1].set_title("(a), (c) ε-limited principal widths √(Hε)/sᵢ", color=ink, loc="left")
    ax[1, 1].set_xlabel("input units")
    for a in (ax[1, 0], ax[1, 1]):
        a.legend(frameon=False, labelcolor=ink)

    a = ax[1, 2]
    a.axis("off")
    lines = [f"H = {H}, k = {k}, {cfg['n_draws']} draws per init,",
             f"{cfg['n_directions']} unit directions per draw", ""]
    for eps, init in runs(cfg):
        r = res[(eps, init)]
        lines.append(f"{init}, ε = {eps:g}: degenerate "
                     f"{int(r['degenerate'].sum())}/{r['degenerate'].size}")
    lines += ["", "(b) r*, widths and |z*| do not depend on ε;", "κ does."]
    a.text(0, 1, "\n".join(lines), va="top", color=muted, family="monospace", fontsize=8.5)
    for a in ax.flat[:5]:
        a.set_ylabel("count")
    fig.suptitle(f"Lens at PyTorch default init, H = {H}, k = {k}", color=ink, x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = os.path.join(ROOT, cfg["out_dir"])
    os.makedirs(out, exist_ok=True)

    import matplotlib
    meta = dict(git_commit=git("rev-parse", "HEAD"),
                # tracked changes outside out_dir (its committed outputs don't count)
                git_dirty=bool(git("status", "--porcelain", "--untracked-files=no", "--",
                                   ".", f":!{cfg['out_dir']}")),
                jax=jax.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__,
                torch=None, python=platform.python_version(),
                x64=bool(jax.config.read("jax_enable_x64")),
                started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                config=cfg)

    rows, arrays = [], {}
    for H, k in cfg["shapes"]:
        res = run_shape(cfg, H, k)
        rows += summary_rows(H, k, res)
        for (eps, init), r in res.items():
            for name, x in r.items():
                arrays[f"H{H}_k{k}/eps{eps:g}/{init}/{name}"] = x
        plot_shape(H, k, res, cfg, os.path.join(out, f"hist_H{H}_k{k}.png"))
        if not all(r["degenerate"].all() for (e, i), r in res.items() if i != "torch_default"):
            print(f"WARNING: a zero-bias draw at H={H}, k={k} is not degenerate")
        print(f"H={H} k={k} done")

    cols = ["H", "k", "eps", "init", "quantity", "n_draws", "n_degenerate", "n", "mean",
            "std", "min", "q05", "q25", "q50", "q75", "q95", "max", "estimate",
            "q50_over_estimate"]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in cols})
    np.savez(os.path.join(out, "draws.npz"), **arrays)
    shutil.copy(args.config, os.path.join(out, "config.yaml"))
    meta["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
