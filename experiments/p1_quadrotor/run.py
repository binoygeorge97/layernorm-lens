"""P-I: quadrotor surrogates (docs/plan.md). Stages:

    python experiments/p1_quadrotor/run.py --config experiments/p1_quadrotor/config.yaml STAGE

- generate, train, analyse, hover: refuse to run unless the annotated tag prereg-p1
  exists and prereg/p1.md matches it (CLAUDE.md: no outcome before the tag). They run
  on quadrotor data.
- benchmark: not gated; times every grid configuration on RANDOM targets (no quadrotor
  data) to estimate the cost of the 40-model grid.

The stage functions take a `Plant` (data.py), so tests drive them with synthetic plants.
Every stage writes its config, the git commit and the package versions next to its
results, and prints the `git add -f` command for its small outputs.
"""

import argparse
import csv
import datetime
import importlib.metadata as md
import json
import os
import platform
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p_ in (ROOT, HERE, os.path.join(ROOT, "experiments", "r6_tdmpc2")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import data as pdata  # noqa: E402
import hover as phover  # noqa: E402
import provenance as pv  # noqa: E402
from lens import analysis as an  # noqa: E402
from lens import geometry as geo  # noqa: E402
from lens import models  # noqa: E402
from lens import train as tr  # noqa: E402

GATED = ("generate", "train", "analyse", "hover")


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #


def versions():
    out = {"python": platform.python_version()}
    for p in ("jax", "jaxlib", "numpy", "scipy", "PyYAML"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def meta(cfg, stage, extra=None):
    return dict(stage=stage, **pv.git_state(), versions=versions(), platform=platform.platform(),
                time=datetime.datetime.now(datetime.timezone.utc).isoformat(), config=cfg, **(extra or {}))


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, float) else v) for k, v in r.items()})


def grid(cfg):
    """[(name, spec, seed)] for the configured grid."""
    g = cfg["grid"]
    out = []
    for arch in g["archs"]:
        for init in g["inits"]:
            for nb in g["n_blocks"]:
                for seed in g["seeds"]:
                    spec = models.SurrogateSpec(arch=arch, n_blocks=int(nb), H=int(g["H"]), k=int(g.get("k", 16)),
                                                n_out=int(g.get("n_out", 12)), eps=float(g["eps"]), init=init,
                                                branch_width=int(g["branch_width"]))
                    out.append((f"{arch}-{init}-b{nb}-s{seed}", spec, int(seed)))
    return out


def save_params(path, p):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in p.items()})


def load_params(path):
    d = np.load(path)
    return {k: jnp.asarray(d[k]) for k in d.files}


def directions_for(ds):
    """d₁ of the training inputs (r6.md's definition) and its half-width D(d₁)."""
    d1, ratio = an.data_d1(ds["train"]["Z"])
    D, _, _ = an.half_width(ds["train"]["Z"], d1)
    return dict(d1=d1), dict(d1=D), ratio


# --------------------------------------------------------------------------- #
# stages                                                                        #
# --------------------------------------------------------------------------- #


def stage_generate(cfg, plant, hx, hu, data_dir, manifest_path):
    s = cfg["sampling"]
    sizes = dict(train=s["n_train"], val=s["n_val"], test=s["n_test"])
    ds = pdata.make_dataset(plant, hx, hu, sizes, int(s["seed"]))
    rows = pdata.save_dataset(ds, data_dir, manifest_path)
    return ds, rows


def train_one(cfg, spec, seed, ds):
    dirs, D, ratio = directions_for(ds)
    p0 = models.init_params(spec, seed)
    t = cfg["training"]
    tcfg = dict(lr=float(t["lr"]), max_steps=int(t["max_steps"]), eval_every=int(t["eval_every"]),
                patience_frac=float(t["patience_frac"]), tol=float(t["tol"]), batch_size=t["batch_size"],
                seed=seed, log_every=int(t["log_every"]))
    data = dict(Z=ds["train"]["Z"], Y=ds["train"]["Y"], Zv=ds["val"]["Z"], Yv=ds["val"]["Y"])
    p, hist, info = tr.train(spec, p0, data, tcfg, directions=dirs, D=D)
    info["d1_eigen_ratio"] = ratio
    return p, hist, info


def history_arrays(hist):
    """Scalars of the training history as columns; per-step z* and widths stacked."""
    scal = [{k: v for k, v in h.items() if not isinstance(v, np.ndarray)} for h in hist]
    logged = [h for h in hist if "z_star" in h]
    arrays = dict(step=np.array([h["step"] for h in logged]),
                  z_star=np.stack([h["z_star"] for h in logged]) if logged else np.zeros((0,)),
                  widths=np.stack([h["widths"] for h in logged]) if logged else np.zeros((0,)))
    return scal, arrays


def stage_train(cfg, ds, ck_dir, res_dir, members=None):
    rows = []
    for name, spec, seed in (members or grid(cfg)):
        p, hist, info = train_one(cfg, spec, seed, ds)
        save_params(os.path.join(ck_dir, f"{name}.npz"), p)
        scal, arrays = history_arrays(hist)
        write_csv(os.path.join(res_dir, "history", f"{name}.csv"), scal)
        np.savez(os.path.join(res_dir, "history", f"{name}.npz"), **arrays)
        rows.append(dict(name=name, arch=spec.arch, init=spec.init, n_blocks=spec.n_blocks, seed=seed,
                         best_step=info["best_step"], best_val=info["best_val"], stopped_step=info["stopped_step"],
                         stopped_early=info["stopped_early"], seconds=info["seconds"], n_params=info["n_params"],
                         d1_eigen_ratio=info["d1_eigen_ratio"]))
    write_csv(os.path.join(res_dir, "train_summary.csv"), rows)
    return rows


def analyse_one(cfg, spec, p, ds):
    a = cfg["analysis"]
    Zt, Jt = ds["test"]["Z"], ds["test"]["J_std"]
    E, b = (np.asarray(m) for m in models.first_layer(spec, p))
    L = geo.lens(E, b, spec.eps)
    err = an.jacobian_error(an.jacobians(spec, p, Zt), Jt, a["jacobian_error"])
    dist, kind = an.lens_distances(L, Zt)
    nf = an.near_far(err, dist, float(a["near_frac"]), float(a["far_frac"]), a["stat"])
    d1, ratio = an.data_d1(ds["train"]["Z"])
    out = dict(degenerate=L.degenerate, kappa=float(L.kappa), norm_z_star=float(np.linalg.norm(L.z_star)),
               dist_kind=kind, d1_eigen_ratio=ratio, err_median=float(np.median(err)),
               **{f"nf_{k}": v for k, v in nf.items()})
    for dname, d in (("d1", d1), ("u_min", an.u_min(L))):
        for k, v in an.direction_sharpness(L, ds["train"]["Z"], d).items():
            out[f"{dname}_{k}"] = v
    if not L.degenerate:
        c1 = a["corollary1"]
        Ztr = ds["train"]["Z"]
        lines = [(Ztr.mean(0), d1)] + [(Ztr[i], d1) for i in np.random.default_rng(int(c1["rng_seed"])).choice(
            len(Ztr), int(c1["n_state_lines"]), replace=False)]
        devs = [an.corollary1_line(L, E, b, x0, d, Ztr, int(c1["n_grid"]), float(c1["t_range"]))["max_rel_dev"]
                for x0, d in lines]
        out["corollary1_max_rel_dev"] = float(max(devs))
        if spec.n_blocks > 1:
            at = an.attenuation(spec, p, L, d1, Ztr, int(a["attenuation"]["n_grid"]), float(a["attenuation"]["t_range"]))
            out.update(attenuation_out=at["attenuation_out"],
                       **{f"attenuation_block{j + 1}": v for j, v in enumerate(at["attenuation_blocks"])})
    return out


def hover_one(cfg, spec, p, ds, plant, hx, hu):
    J_std = np.asarray(jax.jacfwd(lambda z: models.forward(spec, p, z))(jnp.asarray(ds["trim"]["Z"])))
    A_s, B_s = phover.physical_jacobian(J_std, ds["scalers"], plant.n_x)
    A_t, B_t = phover.true_y_jacobians(plant)
    Q, R = phover.bryson(hx, hu)
    r = phover.hover_check(A_s, B_s, A_t, B_t, Q, R, float(cfg["hover"]["sign_rel_threshold"]))
    return {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}


def stage_benchmark(cfg, out_dir, members=None, steps=None):
    """Time every grid configuration on random targets: Z ~ N(0, I_k), Y = a fixed random
    tanh network of Z (12 outputs). No quadrotor data is used."""
    b = cfg["benchmark"]
    rng = np.random.default_rng(int(b["seed"]))
    k, n_out = 16, 12
    W1, W2 = rng.standard_normal((k, 64)) / np.sqrt(k), rng.standard_normal((64, n_out)) / 8.0
    Z, Zv = rng.standard_normal((int(b["n_train"]), k)), rng.standard_normal((int(b["n_val"]), k))
    data = dict(Z=Z, Y=np.tanh(Z @ W1) @ W2, Zv=Zv, Yv=np.tanh(Zv @ W1) @ W2)
    steps = int(steps or b["steps"])
    t = cfg["training"]
    rows, seen = [], set()
    for name, spec, seed in (members or grid(cfg)):
        key = (spec.arch, spec.init, spec.n_blocks)
        if key in seen:
            continue
        seen.add(key)
        p0 = models.init_params(spec, seed)
        base = dict(lr=float(t["lr"]), eval_every=int(t["eval_every"]), patience_frac=10.0, tol=float(t["tol"]),
                    batch_size=t["batch_size"], seed=seed, log_every=0)
        _, _, warm = tr.train(spec, p0, data, dict(base, max_steps=int(t["eval_every"])))  # compile + one chunk
        _, _, info = tr.train(spec, p0, data, dict(base, max_steps=steps))
        per_step = info["seconds"] / info["stopped_step"]
        rows.append(dict(arch=spec.arch, init=spec.init, n_blocks=spec.n_blocks, n_params=info["n_params"],
                         steps=info["stopped_step"], seconds=info["seconds"], seconds_per_step=per_step,
                         first_call_seconds=warm["seconds"]))
    n_models = len(members or grid(cfg))
    per_cfg = {(r["arch"], r["init"], r["n_blocks"]): r["seconds_per_step"] for r in rows}
    total = sum(per_cfg[(s.arch, s.init, s.n_blocks)] for _, s, _ in (members or grid(cfg))) * int(t["max_steps"])
    summary = dict(n_models=n_models, max_steps=int(t["max_steps"]), upper_bound_hours=total / 3600.0,
                   note="Upper bound: every model runs the full budget with no early stop; lens logging excluded.")
    write_csv(os.path.join(out_dir, "benchmark.csv"), rows)
    write_json(os.path.join(out_dir, "meta_benchmark.json"), meta(cfg, "benchmark", dict(summary=summary)))
    return rows, summary


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("stage", choices=list(GATED) + ["benchmark"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    res = os.path.join(ROOT, *cfg["paths"]["results"].split("/"))
    if args.stage == "benchmark":
        rows, summary = stage_benchmark(cfg, os.path.join(res, "benchmark"))
        for r in rows:
            print(r)
        print(summary)
        pv.print_commit_listing([os.path.join(res, "benchmark", f) for f in ("benchmark.csv", "meta_benchmark.json")])
        return
    pre = cfg["prereg"]
    try:
        pv.require_prereg(tags=[(pre["tag"], pre["file"])])
    except pv.ProvenanceError as e:
        sys.exit(str(e))
    if pv.git_state()["git_dirty"]:
        sys.exit("refusing to run: the working tree has uncommitted changes to tracked files.")
    plant, _ = pdata.quadrotor_plant(cfg["plant"])
    hx, hu = pdata.quadrotor_half_widths(cfg["sampling"]["half_widths"], plant.u0)
    data_dir = os.path.join(ROOT, *cfg["paths"]["data"].split("/"))
    manifest = os.path.join(ROOT, *cfg["paths"]["manifest"].split("/"))
    ck = os.path.join(ROOT, *cfg["paths"]["checkpoints"].split("/"))
    if args.stage == "generate":
        _, rows = stage_generate(cfg, plant, hx, hu, data_dir, manifest)
        write_json(os.path.join(res, "meta_generate.json"), meta(cfg, "generate", dict(files=rows)))
        pv.print_commit_listing([manifest, os.path.join(res, "meta_generate.json")])
        return
    ds = pdata.load_dataset(data_dir, manifest)
    if args.stage == "train":
        stage_train(cfg, ds, ck, os.path.join(res, "train"))
        write_json(os.path.join(res, "train", "meta_train.json"), meta(cfg, "train"))
        return
    rows = []
    for name, spec, _ in grid(cfg):
        p = load_params(os.path.join(ck, f"{name}.npz"))
        r = analyse_one(cfg, spec, p, ds) if args.stage == "analyse" else hover_one(cfg, spec, p, ds, plant, hx, hu)
        rows.append(dict(name=name, **r))
    write_csv(os.path.join(res, f"{args.stage}.csv"), rows)
    write_json(os.path.join(res, f"meta_{args.stage}.json"), meta(cfg, args.stage))
    pv.print_commit_listing([os.path.join(res, f"{args.stage}.csv"), os.path.join(res, f"meta_{args.stage}.json")])


if __name__ == "__main__":
    main()
