"""P-I: quadrotor surrogates (docs/plan.md; prereg/drafts/p1-draft.md). Stages:

    python experiments/p1_quadrotor/run.py --config experiments/p1_quadrotor/config.yaml STAGE
        [--index I] [--out-root DIR] [--data-dir DIR] [--of train|train_long]

- generate, train, train_long, gather, analyse, analyse_long, hover, p5, predictions: refuse to run
  unless the annotated tag prereg-p1 exists and prereg/p1.md matches it (CLAUDE.md: no
  outcome before the tag), and the working tree is clean. They run on quadrotor data.
  - train: the 40-model grid (early stopping). train_long: the long-budget subset
    (`long_budget` in the config; early stopping disabled, parameter snapshots).
  - `--index I` runs only the I-th member of the stage's grid (one Slurm array task,
    slurm/p1_array.slurm); each run writes its summary JSON and a SHA-256 manifest of
    its outputs, and `gather --of STAGE` verifies and merges them.
  - `--out-root DIR` puts data, checkpoints and results under DIR (e.g. $SCRATCH); the
    data manifest is always the committed one in the repository.
  - analyse: predictions 1–4's quantities on the 40 models; analyse_long: prediction 1's
    statistic and the lens at every snapshot of the long runs (the race of the draft);
    hover: the hover check, H1's per-model flag and the trim check; p5: prediction 5
    from the lens logs of both stages; predictions: every pre-registered rule
    (p1_rules.py) applied to those outputs, with G2.
- benchmark: not gated; times every grid configuration on RANDOM targets (no quadrotor
  data) to estimate the cost of the grid and the long-budget subset.
- init_lens: not gated; the initial lens of every seed from initialisation alone, on
  synthetic i.i.d. uniform inputs (no quadrotor data): the p1 draft's initial values.
- trims: not gated; samples the hover check's 200 steady-flight trims with a fixed seed
  and verifies them on the simulator (no surrogate). Run once before any training; a
  rerun must reproduce the committed file byte for byte.

The stage functions take a `Plant` (p1_data.py), so tests drive them with synthetic plants.
Every stage writes its config, the git commit and the package versions next to its
results, and prints the `git add -f` command for its small outputs.
"""

import argparse
import csv
import datetime
import glob
import importlib.metadata as md
import json
import os
import platform
import socket
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p_ in (ROOT, HERE, os.path.join(ROOT, "experiments", "r6_tdmpc2")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import p1_data as pdata  # noqa: E402
import p1_hover as phover  # noqa: E402
import p1_rules as rules  # noqa: E402
import provenance as pv  # noqa: E402
from lens import analysis as an  # noqa: E402
from lens import geometry as geo  # noqa: E402
from lens import models  # noqa: E402
from lens import train as tr  # noqa: E402

GATED = ("generate", "train", "train_long", "gather", "analyse", "analyse_long", "hover", "p5", "predictions")
GRIDS = {"train": "grid", "train_long": "long_budget"}


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


def slurm_env():
    """The Slurm job's identity, if any (None outside Slurm), and the host."""
    keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST", "SLURM_CLUSTER_NAME")
    return dict(host=socket.gethostname(), **{k: os.environ.get(k) for k in keys})


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


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grid(cfg, which="grid"):
    """[(name, spec, seed)] for the configured grid ("grid": the 40 models) or for the
    long-budget subset ("long_budget": its archs, inits, n_blocks and seeds, with the
    grid's H, k, ε and branch width)."""
    g = dict(cfg["grid"])
    if which != "grid":
        g.update({k: cfg[which][k] for k in ("archs", "inits", "n_blocks", "seeds")})
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


def paths(cfg, out_root=None, data_dir=None):
    """Data, checkpoints and results under `out_root` (default: the repository); the data
    manifest always in the repository (it is committed and verifies the data anywhere)."""
    base = out_root or ROOT
    j = lambda root, key: os.path.join(root, *cfg["paths"][key].split("/"))  # noqa: E731
    return dict(base=base, data=data_dir or j(base, "data"), manifest=j(ROOT, "manifest"),
                ck=j(base, "checkpoints"), res=j(base, "results"))


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


def train_cfg(cfg, seed, long=False):
    """lens/train.py's config for one run: `training`, or for the long-budget subset its
    budget and logging with early stopping disabled."""
    t = dict(cfg["training"])
    if long:
        lb = cfg["long_budget"]
        t.update(max_steps=lb["max_steps"], log_every=lb["log_every"], patience_frac=None)
    return dict(lr=float(t["lr"]), max_steps=int(t["max_steps"]), eval_every=int(t["eval_every"]),
                patience_frac=None if t["patience_frac"] is None else float(t["patience_frac"]),
                tol=float(t["tol"]), batch_size=None if t["batch_size"] is None else int(t["batch_size"]),
                seed=seed, log_every=int(t["log_every"]))


def initial_r_star_median(spec, seed, n_dir=64, rng_seed=0):
    """Prediction 2's initial r*: the median over n_dir random unit directions of r*(d)
    for the initial first layer (theory.md convention 4; the initial-lens check used 64
    directions per draw). For a degenerate lens, the ε-limited √(Hε)/‖A d‖ instead."""
    E, b = (np.asarray(m) for m in models.first_layer(spec, models.init_params(spec, seed)))
    L = geo.lens(E, b, spec.eps)
    d = np.random.default_rng(rng_seed).standard_normal((n_dir, spec.k))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return float(np.median([geo.line(L, dj)["r_eff" if L.degenerate else "r_star"] for dj in d]))


# --------------------------------------------------------------------------- #
# stages                                                                        #
# --------------------------------------------------------------------------- #


def stage_generate(cfg, plant, hx, hu, data_dir, manifest_path):
    s = cfg["sampling"]
    sizes = dict(train=s["n_train"], val=s["n_val"], test=s["n_test"])
    ds = pdata.make_dataset(plant, hx, hu, sizes, int(s["seed"]))
    rows = pdata.save_dataset(ds, data_dir, manifest_path)
    return ds, rows


def train_one(cfg, spec, seed, ds, long=False, on_eval=None):
    dirs, D, ratio = directions_for(ds)
    p0 = models.init_params(spec, seed)
    data = dict(Z=ds["train"]["Z"], Y=ds["train"]["Y"], Zv=ds["val"]["Z"], Yv=ds["val"]["Y"])
    p, hist, info = tr.train(spec, p0, data, train_cfg(cfg, seed, long), directions=dirs, D=D,
                             lens_Z=ds["train"]["Z"], on_eval=on_eval)
    info["d1_eigen_ratio"] = ratio
    return p, hist, info


def history_arrays(hist):
    """Scalars of the training history as columns; per-step z*, widths and u_min stacked."""
    scal = [{k: v for k, v in h.items() if not isinstance(v, np.ndarray)} for h in hist]
    logged = [h for h in hist if "z_star" in h]
    stack = lambda key: np.stack([h[key] for h in logged]) if logged and key in logged[0] else np.zeros((0,))  # noqa: E731
    arrays = dict(step=np.array([h["step"] for h in logged]), z_star=stack("z_star"), widths=stack("widths"),
                  u_min=stack("u_min"))
    return scal, arrays


def stage_train(cfg, ds, ck_dir, res_dir, members=None, long=False, base=None, summary=True):
    """Train each member and save: its parameters (the best held-out MSE's; for a long run
    also the last, and snapshots every `long_budget.snapshot_every` steps), its history,
    a per-run JSON (summary row, commit, versions, config, Slurm identity) and a per-run
    SHA-256 manifest of those files, with paths relative to `base`. `summary` writes
    train_summary.csv for the members (off for a single array task; see stage_gather)."""
    base = base or ROOT
    stage = "train_long" if long else "train"
    rows = []
    for name, spec, seed in (members or grid(cfg, GRIDS[stage])):
        files, last = [], {}
        on_eval = None
        if long:
            every = int(cfg["long_budget"]["snapshot_every"])

            def on_eval(step, p, name=name, every=every):
                last.update(step=step, p=p)
                if step % every == 0:
                    path = os.path.join(ck_dir, name, f"step{step:07d}.npz")
                    save_params(path, p)
                    files.append(path)
        p, hist, info = train_one(cfg, spec, seed, ds, long, on_eval)
        files.append(os.path.join(ck_dir, f"{name}.npz"))
        save_params(files[-1], p)
        if long:
            files.append(os.path.join(ck_dir, f"{name}-last.npz"))
            save_params(files[-1], last["p"])
        scal, arrays = history_arrays(hist)
        hcsv, hnpz = (os.path.join(res_dir, "history", f"{name}.{e}") for e in ("csv", "npz"))
        write_csv(hcsv, scal)
        np.savez(hnpz, **arrays)
        row = dict(name=name, stage=stage, arch=spec.arch, init=spec.init, n_blocks=spec.n_blocks, seed=seed,
                   best_step=info["best_step"], best_val=info["best_val"], stopped_step=info["stopped_step"],
                   stopped_early=info["stopped_early"], seconds=info["seconds"], n_params=info["n_params"],
                   d1_eigen_ratio=info["d1_eigen_ratio"])
        rjson = os.path.join(res_dir, "runs", f"{name}.json")
        write_json(rjson, meta(cfg, stage, dict(run=row, slurm=slurm_env(), train_info=info)))
        pv.write_manifest(os.path.join(res_dir, "runs", f"{name}_manifest.csv"), files + [hcsv, hnpz, rjson], base)
        rows.append(row)
    if summary:
        write_csv(os.path.join(res_dir, "train_summary.csv"), rows)
    return rows


def stage_gather(res_dir, base, members, require_clean=True):
    """Verify and merge the per-run outputs of one training stage: every member's JSON
    and manifest must exist, every manifest file must exist under `base` with its
    SHA-256, and all runs must share one git commit, clean unless `require_clean` is
    False (tests only; `main` always requires it). Writes train_summary.csv and
    outputs_manifest.csv (run, file, bytes, sha256). Raises pv.ProvenanceError on any
    problem, listing them all."""
    rows, mrows, problems, commits = [], [], [], set()
    for name, _, _ in members:
        rj, mf = (os.path.join(res_dir, "runs", f"{name}{s}") for s in (".json", "_manifest.csv"))
        if not (os.path.exists(rj) and os.path.exists(mf)):
            problems.append(f"{name}: run JSON or manifest missing")
            continue
        for r in pv.read_manifest(mf):
            path = os.path.join(base, *r["file"].replace("\\", "/").split("/"))
            if not os.path.exists(path):
                problems.append(f"{name}: {r['file']} missing")
            elif pv.sha256(path) != r["sha256"]:
                problems.append(f"{name}: {r['file']} SHA-256 mismatch")
            mrows.append(dict(run=name, file=r["file"].replace("\\", "/"), bytes=r["bytes"], sha256=r["sha256"]))
        with open(rj, encoding="utf-8") as f:
            m = json.load(f)
        commits.add((m["git_commit"], m["git_dirty"]))
        rows.append(m["run"])
    if len(commits) > 1:
        problems.append(f"runs differ in commit: {sorted(commits)}")
    if require_clean and any(dirty for _, dirty in commits):
        problems.append(f"a run ran on a dirty tree: {sorted(commits)}")
    if problems:
        raise pv.ProvenanceError("gather: " + "; ".join(problems))
    write_csv(os.path.join(res_dir, "train_summary.csv"), rows)
    write_csv(os.path.join(res_dir, "outputs_manifest.csv"), mrows)
    return rows, mrows


def analyse_one(cfg, spec, p, ds, seed=None):
    """Predictions 1–4's quantities for one trained model (p1 draft):
    (1) the near/far Jacobian-error ratio and ‖z* − μ‖ (μ the training inputs' mean);
    (2) the initial median r* (if `seed` is given); (3) sharpness and coverage along d₁ and
    u_min, and the affected-data fraction (test states whose error exceeds
    `affected_factor` times the far set's median); (4) attenuation along d₁ and u_min for
    stacks. Also the Corollary 1 deviations (a pipeline check)."""
    a = cfg["analysis"]
    Zt, Jt = ds["test"]["Z"], ds["test"]["J_std"]
    Ztr = ds["train"]["Z"]
    E, b = (np.asarray(m) for m in models.first_layer(spec, p))
    L = geo.lens(E, b, spec.eps)
    err = an.jacobian_error(an.jacobians(spec, p, Zt), Jt, a["jacobian_error"])
    dist, kind = an.lens_distances(L, Zt)
    nf = an.near_far(err, dist, float(a["near_frac"]), float(a["far_frac"]), a["stat"])
    d1, ratio = an.data_d1(Ztr)
    um = an.u_min(L)
    out = dict(degenerate=L.degenerate, kappa=float(L.kappa), norm_z_star=float(np.linalg.norm(L.z_star)),
               z_star_to_mean=float(np.linalg.norm(L.z_star - Ztr.mean(0))),
               dist_kind=kind, d1_eigen_ratio=ratio, err_median=float(np.median(err)),
               affected_frac=float(np.mean(err > float(a["affected_factor"]) * nf["far"])),
               **{f"nf_{k}": v for k, v in nf.items()})
    if seed is not None:
        ri = a["r_star_init"]
        out["r_star_init_median"] = initial_r_star_median(spec, seed, int(ri["n_directions"]), int(ri["rng_seed"]))
    for dname, d in (("d1", d1), ("u_min", um)):
        for k, v in an.direction_sharpness(L, Ztr, d).items():
            out[f"{dname}_{k}"] = v
    if not L.degenerate:
        c1 = a["corollary1"]
        lines = [(Ztr.mean(0), d1)] + [(Ztr[i], d1) for i in np.random.default_rng(int(c1["rng_seed"])).choice(
            len(Ztr), int(c1["n_state_lines"]), replace=False)]
        devs = [an.corollary1_line(L, E, b, x0, d, Ztr, int(c1["n_grid"]), float(c1["t_range"]))["max_rel_dev"]
                for x0, d in lines]
        out["corollary1_max_rel_dev"] = float(max(devs))
    if spec.n_blocks > 1:
        for dname, d in (("u_min", um), ("d1", d1)):
            sr = an.attenuation_S(spec, p, L, d)
            out.update({f"p4_S_out_{dname}": sr["S_out"], f"p4_S_block1_{dname}": sr["S_block1"],
                        f"p4_ratio_{dname}": sr["ratio"]})
        for dname, d in (("d1", d1), ("u_min", um)):
            at = an.attenuation(spec, p, L, d, Ztr, int(a["attenuation"]["n_grid"]), float(a["attenuation"]["t_range"]))
            pre = "" if dname == "d1" else "u_min_"
            out.update({f"{pre}attenuation_out": at["attenuation_out"],
                        **{f"{pre}attenuation_block{j + 1}": v for j, v in enumerate(at["attenuation_blocks"])}})
    return out


def p3_spearman(rows, scores=("u_min_sharpness", "d1_sharpness", "u_min_coverage", "d1_coverage"),
                target="affected_frac"):
    """Prediction 3: Spearman's rank correlation, over the models, between each
    weight-based score and the affected-data fraction."""
    y = np.array([float(r[target]) for r in rows])
    return {s: float(stats.spearmanr(np.array([float(r[s]) for r in rows]), y).statistic) for s in scores}


def race_point(cfg, spec, p, ds):
    """Prediction 1's statistic and the lens for one parameter snapshot (analyse_long)."""
    a = cfg["analysis"]
    Zt, Ztr = ds["test"]["Z"], ds["train"]["Z"]
    E, b = (np.asarray(m) for m in models.first_layer(spec, p))
    L = geo.lens(E, b, spec.eps)
    err = an.jacobian_error(an.jacobians(spec, p, Zt), ds["test"]["J_std"], a["jacobian_error"])
    dist, kind = an.lens_distances(L, Zt)
    nf = an.near_far(err, dist, float(a["near_frac"]), float(a["far_frac"]), a["stat"])
    um = an.u_min(L)
    s = an.direction_sharpness(L, Ztr, um)
    return dict(degenerate=L.degenerate, kappa=float(L.kappa), dist_kind=kind, nf_ratio=nf["ratio"],
                err_median=float(np.median(err)), z_star_to_mean=float(np.linalg.norm(L.z_star - Ztr.mean(0))),
                u_min_r_eff_over_D=s["r_eff"] / s["D"])


def stage_analyse_long(cfg, ds, ck_dir, members):
    rows = []
    for name, spec, _ in members:
        for path in sorted(glob.glob(os.path.join(ck_dir, name, "step*.npz"))):
            step = int(os.path.basename(path)[4:-4])
            rows.append(dict(name=name, step=step, **race_point(cfg, spec, load_params(path), ds)))
    return rows


# --------------------------------------------------------------------------- #
# prediction 5                                                                  #
# --------------------------------------------------------------------------- #


def p5_run(steps, ratio, kappa, end_step, fold=10.0, floor=0.1, kappa_max=0.1):
    """Prediction 5 for one run's lens log (p1 draft): along u_min, the ratio
    r_eff(u_min)/D(u_min) at `end_step` is at least `fold` times its value at step 0 and
    at least `floor`, and κ at `end_step` is at most `kappa_max` (a degenerate lens has
    κ = ∞ and fails). Also returns the largest ratio up to end_step."""
    steps, ratio, kappa = (np.asarray(v, np.float64) for v in (steps, ratio, kappa))
    if not (np.any(steps == 0) and np.any(steps == end_step)):
        raise ValueError(f"the lens log must include step 0 and the end step {end_step} (log at every evaluation)")
    i0, ie = int(np.flatnonzero(steps == 0)[0]), int(np.flatnonzero(steps == end_step)[0])
    r0, re, ke = float(ratio[i0]), float(ratio[ie]), float(kappa[ie])
    return dict(end_step=int(end_step), ratio_init=r0, ratio_end=re, fold_end=re / r0, kappa_end=ke,
                ratio_max=float(np.max(ratio[:ie + 1])),
                passes=bool(re >= fold * r0 and re >= floor and ke <= kappa_max))


def p5_cells(results, keys, min_pass=4):
    """Count passes per cell (results grouped by `keys`); a cell holds if at least
    `min_pass` of its runs pass."""
    cells = {}
    for r in results:
        cells.setdefault(tuple(r[k] for k in keys), []).append(bool(r["passes"]))
    return [dict(**dict(zip(keys, c)), n=len(v), n_pass=int(sum(v)), holds=sum(v) >= min_pass)
            for c, v in sorted(cells.items())]


def stage_p5(cfg, res, which):
    """Prediction 5 on one training stage's lens logs. End: the returned (best held-out
    MSE) parameters for `train`, the last step for `train_long`."""
    q = cfg["p5"]
    rd = os.path.join(res, which)
    summ = {r["name"]: r for r in read_csv(os.path.join(rd, "train_summary.csv"))}
    out = []
    for name, spec, seed in grid(cfg, GRIDS[which]):
        h = [r for r in read_csv(os.path.join(rd, "history", f"{name}.csv")) if r.get("kappa", "") != ""]
        end = int(summ[name]["best_step"] if which == "train" else summ[name]["stopped_step"])
        r = p5_run([int(x["step"]) for x in h], [float(x["r_eff_over_D_u_min"]) for x in h],
                   [float(x["kappa"]) for x in h], end, float(q["fold"]), float(q["floor"]), float(q["kappa_max"]))
        out.append(dict(name=name, stage=which, arch=spec.arch, init=spec.init, n_blocks=spec.n_blocks, seed=seed, **r))
    keys = ("arch", "n_blocks") if which == "train" else ("arch",)
    return out, p5_cells([r for r in out if r["init"] == "zero_bias"], keys, int(q["min_seeds"]))


def hover_one(cfg, spec, p, ds, plant, hx, hu, trims=None, truth=None):
    """The hover check at hover (with H1's per-model flag), and, if trims are given, the
    trim check: per-trim rows, summarised per model by Spearman correlations of the
    variation error and of A's and B's errors with the trims' lens distance, and their
    medians. Returns (summary, trim rows)."""
    J_std = np.asarray(jax.jacfwd(lambda z: models.forward(spec, p, z))(jnp.asarray(ds["trim"]["Z"])))
    A_s, B_s = phover.physical_jacobian(J_std, ds["scalers"], plant.n_x)
    A_t, B_t = phover.true_y_jacobians(plant)
    Q, R = phover.bryson(hx, hu)
    thr = float(cfg["hover"]["sign_rel_threshold"])
    r = phover.hover_check(A_s, B_s, A_t, B_t, Q, R, thr)
    out = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
    if trims is None:
        return out, []
    E, b = (np.asarray(m) for m in models.first_layer(spec, p))
    L = geo.lens(E, b, spec.eps)
    Jb = jax.jit(jax.vmap(jax.jacfwd(lambda z: models.forward(spec, p, z))))
    trows = phover.trim_check(lambda Z: Jb(jnp.asarray(Z, jnp.float64)), ds["scalers"], plant, trims, truth,
                              ds["trim"]["Z"], lambda Z: an.lens_distances(L, Z)[0], thr)
    dist = np.array([t["lens_distance"] for t in trows])
    for key in ("variation_err", "rel_err_A", "rel_err_B"):
        v = np.array([t[key] for t in trows])
        out[f"trims_{key}_median"] = float(np.median(v))
        out[f"trims_spearman_dist_{key}"] = float(stats.spearmanr(dist, v).statistic)
    out["trims_sign_agree_min"] = float(min(min(t["sign_agree_A"], t["sign_agree_B"]) for t in trows))
    return out, trows


def stage_trims(cfg, plant, hx, out_dir):
    """Sample the trims (hover.trims: n, seed), verify each is a trim of the simulator and
    inside the training box, and write trims.csv, its SHA-256 manifest and meta. If
    trims.csv exists, the regenerated bytes must match it (regeneration identity)."""
    t = cfg["hover"]["trims"]
    X = phover.sample_trims(plant, hx, int(t["n"]), int(t["seed"]))
    inside, dev = phover.check_trims(plant, X, hx)
    if not inside or dev > float(t["step_tol"]):
        raise pv.ProvenanceError(f"trims: inside the box {inside}, max step deviation {dev}")
    rows = [dict(trim=i, **{f"x{j}": repr(float(v)) for j, v in enumerate(x)}) for i, x in enumerate(X)]
    path = os.path.join(out_dir, "trims.csv")
    new = os.path.join(out_dir, "trims.csv.new")
    write_csv(new, rows)
    if os.path.exists(path):
        with open(path, "rb") as f1, open(new, "rb") as f2:
            same = f1.read() == f2.read()
        os.remove(new)
        if not same:
            raise pv.ProvenanceError(f"trims: the regenerated trims differ from {path}")
    else:
        os.replace(new, path)
    pv.write_manifest(os.path.join(out_dir, "trims_manifest.csv"), [path], out_dir)
    write_json(os.path.join(out_dir, "meta_trims.json"),
               meta(cfg, "trims", dict(n=len(X), inside_box=inside, max_step_deviation=dev)))
    return X, dev


def load_trims(cfg, res):
    """The committed trims, after verifying trims.csv against its manifest."""
    d = os.path.join(res, cfg["hover"]["trims"]["out"])
    m = pv.read_manifest(os.path.join(d, "trims_manifest.csv"))
    pv.verify_sha256(os.path.join(d, "trims.csv"), m[0]["sha256"], "trims.csv")
    rows = read_csv(os.path.join(d, "trims.csv"))
    n_x = len([k for k in rows[0] if k.startswith("x")])
    return np.array([[float(r[f"x{j}"]) for j in range(n_x)] for r in rows])


def stage_predictions(cfg, res):
    """Apply every pre-registered rule (p1_rules.py) to the stage outputs."""
    q = cfg["predictions"]
    an_rows = read_csv(os.path.join(res, "analyse.csv"))
    hv_rows = read_csv(os.path.join(res, "hover.csv"))
    p1c = rules.p1(an_rows, q)
    out = dict(p1=p1c, g2=rules.g2(p1c, q), p2=rules.p2(an_rows, q), p3=rules.p3(an_rows, q),
               p4=rules.p4(an_rows, q), h1=rules.h1(hv_rows, q),
               p5=rules.p5(read_csv(os.path.join(res, "p5_train_long_cells.csv")),
                           read_csv(os.path.join(res, "p5_train_cells.csv"))),
               race=rules.race(p1c, read_csv(os.path.join(res, "p5_train.csv"))))
    return out


# --------------------------------------------------------------------------- #
# initial values (initialisation alone; no quadrotor data)                      #
# --------------------------------------------------------------------------- #


def stage_init_lens(cfg, out_dir):
    """Initial values from initialisation alone (no quadrotor data, not gated): the lens of
    the initial first layer for seeds 0 … n_seeds − 1 of each initialisation. At a given
    seed the first layer (E, b) is the first draw for both architectures and both depths,
    so one lens per (init, seed) covers all four cells.

    Inputs: the sampling design draws every input coordinate independently and uniformly
    in its box, so the standardised inputs are i.i.d. U(−√3, √3) up to sampling noise;
    `n_inputs` such draws, z-scored by their own mean and standard deviation, stand in for
    the training inputs. Per seed: κ, ‖z*‖, ‖z* − μ‖, and along u_min (the lens's
    narrowest principal direction) and d₁ (the inputs' principal direction): D, r_eff and
    r_eff/D; and prediction 2's median r*(d) over the analysis config's random directions
    (ε-limited for a degenerate lens)."""
    c = cfg["init_lens"]
    rng = np.random.default_rng(int(c["input_seed"]))
    g = cfg["grid"]
    k = int(g.get("k", 16))
    Z = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), (int(c["n_inputs"]), k))
    Z = (Z - Z.mean(0)) / Z.std(0)
    d1, d1_ratio = an.data_d1(Z)
    ri = cfg["analysis"]["r_star_init"]
    rows = []
    for init in g["inits"]:
        spec = models.SurrogateSpec(arch=g["archs"][0], n_blocks=1, H=int(g["H"]), k=k, eps=float(g["eps"]),
                                    init=init, branch_width=int(g["branch_width"]))
        for seed in range(int(c["n_seeds"])):
            E, b = (np.asarray(m) for m in models.first_layer(spec, models.init_params(spec, seed)))
            L = geo.lens(E, b, spec.eps)
            row = dict(init=init, seed=seed, degenerate=L.degenerate, kappa=float(L.kappa),
                       norm_z_star=float(np.linalg.norm(L.z_star)),
                       z_star_to_mean=float(np.linalg.norm(L.z_star - Z.mean(0))),
                       r_star_init_median=initial_r_star_median(spec, seed, int(ri["n_directions"]),
                                                                int(ri["rng_seed"])))
            for name, d in (("u_min", an.u_min(L)), ("d1", d1)):
                s = an.direction_sharpness(L, Z, d)
                row.update({f"D_{name}": s["D"], f"r_eff_{name}": s["r_eff"], f"r_eff_over_D_{name}": s["r_eff"] / s["D"]})
            rows.append(row)
    qs = (5, 50, 95)
    summary = []
    for init in g["inits"]:
        rr = [r for r in rows if r["init"] == init]
        for key in ("r_eff_over_D_u_min", "r_eff_over_D_d1", "r_star_init_median", "kappa", "norm_z_star",
                    "D_u_min", "D_d1"):
            v = np.array([r[key] for r in rr], np.float64)
            fin = v[np.isfinite(v)]
            summary.append(dict(init=init, quantity=key, n=len(v), n_finite=len(fin),
                                **{f"q{q}": float(np.percentile(fin, q)) if len(fin) else np.nan for q in qs},
                                **{f"seed{s}": float(v[s]) for s in g["seeds"]}))
    write_csv(os.path.join(out_dir, "init_lens_summary.csv"), summary)
    write_json(os.path.join(out_dir, "meta_init_lens.json"),
               meta(cfg, "init_lens", dict(d1_eigen_ratio=d1_ratio, n_rows=len(rows))))
    return rows, summary


# --------------------------------------------------------------------------- #
# benchmark (random targets only)                                               #
# --------------------------------------------------------------------------- #


def stage_benchmark(cfg, out_dir, members=None, steps=None):
    """Time every grid configuration on random targets: Z ~ N(0, I_k), Y = a fixed random
    tanh network of Z (12 outputs). No quadrotor data is used. Also times one lens record
    (d₁ and u_min) per configuration. Summary: upper bounds for the grid at its budget
    and for the long-budget subset at its budget (no early stop; lens logging included
    at the configured interval)."""
    b = cfg["benchmark"]
    rng = np.random.default_rng(int(b["seed"]))
    k, n_out = 16, 12
    W1, W2 = rng.standard_normal((k, 64)) / np.sqrt(k), rng.standard_normal((64, n_out)) / 8.0
    Z, Zv = rng.standard_normal((int(b["n_train"]), k)), rng.standard_normal((int(b["n_val"]), k))
    data = dict(Z=Z, Y=np.tanh(Z @ W1) @ W2, Zv=Zv, Yv=np.tanh(Zv @ W1) @ W2)
    d1 = an.data_d1(Z)[0]
    steps = int(steps or b["steps"])
    t = cfg["training"]
    rows, seen = [], set()
    for name, spec, seed in (members or grid(cfg)):
        key = (spec.arch, spec.init, spec.n_blocks)
        if key in seen:
            continue
        seen.add(key)
        p0 = models.init_params(spec, seed)
        base = dict(lr=float(t["lr"]), eval_every=int(t["eval_every"]), patience_frac=None, tol=float(t["tol"]),
                    batch_size=t["batch_size"], seed=seed, log_every=0)
        _, _, warm = tr.train(spec, p0, data, dict(base, max_steps=int(t["eval_every"])))  # compile + one chunk
        _, _, info = tr.train(spec, p0, data, dict(base, max_steps=steps))
        per_step = info["seconds"] / info["stopped_step"]
        tr.lens_record(spec, p0, dict(d1=d1), dict(d1=1.0), Z)  # warm-up
        t0 = time.time()
        for _ in range(3):
            tr.lens_record(spec, p0, dict(d1=d1), dict(d1=1.0), Z)
        rows.append(dict(arch=spec.arch, init=spec.init, n_blocks=spec.n_blocks, n_params=info["n_params"],
                         batch_size=t["batch_size"], steps=info["stopped_step"], seconds=info["seconds"],
                         seconds_per_step=per_step, first_call_seconds=warm["seconds"],
                         lens_record_seconds=(time.time() - t0) / 3.0))
    per_cfg = {(r["arch"], r["init"], r["n_blocks"]): r for r in rows}

    def bound(mem, max_steps, log_every):
        return sum(per_cfg[(s.arch, s.init, s.n_blocks)]["seconds_per_step"] * max_steps
                   + per_cfg[(s.arch, s.init, s.n_blocks)]["lens_record_seconds"] * (max_steps // log_every + 1)
                   for _, s, _ in mem) / 3600.0

    mem = members or grid(cfg)
    summary = dict(n_models=len(mem), batch_size=t["batch_size"], max_steps=int(t["max_steps"]),
                   upper_bound_hours=bound(mem, int(t["max_steps"]), int(t["log_every"])),
                   note="Upper bound: every model runs its full budget with no early stop; lens logging included.")
    if "long_budget" in cfg and members is None:
        lb = cfg["long_budget"]
        lmem = grid(cfg, "long_budget")
        summary.update(long_n_models=len(lmem), long_max_steps=int(lb["max_steps"]),
                       long_hours=bound(lmem, int(lb["max_steps"]), int(lb["log_every"])))
    write_csv(os.path.join(out_dir, "benchmark.csv"), rows)
    write_json(os.path.join(out_dir, "meta_benchmark.json"), meta(cfg, "benchmark", dict(summary=summary)))
    return rows, summary


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("stage", choices=list(GATED) + ["benchmark", "init_lens", "trims"])
    ap.add_argument("--index", type=int, help="run only this member of the stage's grid (a Slurm array task)")
    ap.add_argument("--out-root", help="data, checkpoints and results under this directory (e.g. $SCRATCH)")
    ap.add_argument("--data-dir", help="the data directory, if not <out-root>/<paths.data>")
    ap.add_argument("--of", choices=list(GRIDS), default="train", help="gather / p5: which training stage")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    P = paths(cfg, args.out_root, args.data_dir)
    res = P["res"]
    if args.stage == "benchmark":
        out = os.path.join(res, cfg["benchmark"]["out"])
        rows, summary = stage_benchmark(cfg, out)
        for r in rows:
            print(r)
        print(summary)
        pv.print_commit_listing([os.path.join(out, f) for f in ("benchmark.csv", "meta_benchmark.json")])
        return
    if args.stage == "trims":
        plant, _ = pdata.quadrotor_plant(cfg["plant"])
        hx, _ = pdata.quadrotor_half_widths(cfg["sampling"]["half_widths"], plant.u0)
        out = os.path.join(res, cfg["hover"]["trims"]["out"])
        try:
            X, dev = stage_trims(cfg, plant, hx, out)
        except pv.ProvenanceError as e:
            sys.exit(str(e))
        print(f"{len(X)} trims, inside the box, max step deviation {dev:.3e}")
        pv.print_commit_listing([os.path.join(out, f) for f in ("trims.csv", "trims_manifest.csv", "meta_trims.json")])
        return
    if args.stage == "init_lens":
        out = os.path.join(res, cfg["init_lens"]["out"])
        _, summary = stage_init_lens(cfg, out)
        for r in summary:
            print(r)
        pv.print_commit_listing([os.path.join(out, f) for f in ("init_lens_summary.csv", "meta_init_lens.json")])
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
    if args.stage == "generate":
        _, rows = stage_generate(cfg, plant, hx, hu, P["data"], P["manifest"])
        write_json(os.path.join(res, "meta_generate.json"), meta(cfg, "generate", dict(files=rows)))
        pv.print_commit_listing([P["manifest"], os.path.join(res, "meta_generate.json")])
        return
    if args.stage == "gather":
        rd = os.path.join(res, args.of)
        try:
            stage_gather(rd, P["base"], grid(cfg, GRIDS[args.of]))
        except pv.ProvenanceError as e:
            sys.exit(str(e))
        print(f"gather {args.of}: every run present, every SHA-256 verified, one clean commit")
        return
    if args.stage == "predictions":
        out = stage_predictions(cfg, res)
        write_json(os.path.join(res, "predictions.json"), meta(cfg, "predictions", dict(results=out)))
        print(json.dumps(out, indent=1, default=_json))
        pv.print_commit_listing([os.path.join(res, "predictions.json")])
        return
    if args.stage == "p5":
        runs, cells = stage_p5(cfg, res, args.of)
        write_csv(os.path.join(res, f"p5_{args.of}.csv"), runs)
        write_csv(os.path.join(res, f"p5_{args.of}_cells.csv"), cells)
        write_json(os.path.join(res, f"meta_p5_{args.of}.json"), meta(cfg, "p5", dict(of=args.of)))
        pv.print_commit_listing([os.path.join(res, f) for f in (f"p5_{args.of}.csv", f"p5_{args.of}_cells.csv",
                                                                 f"meta_p5_{args.of}.json")])
        return
    ds = pdata.load_dataset(P["data"], P["manifest"])
    if args.stage in GRIDS:
        long = args.stage == "train_long"
        members = grid(cfg, GRIDS[args.stage])
        if args.index is not None:
            members = [members[args.index]]
        ck = os.path.join(P["ck"], "long") if long else P["ck"]
        stage_train(cfg, ds, ck, os.path.join(res, args.stage), members, long, P["base"], summary=args.index is None)
        if args.index is None:
            write_json(os.path.join(res, args.stage, f"meta_{args.stage}.json"), meta(cfg, args.stage))
        return
    if args.stage == "analyse_long":
        rows = stage_analyse_long(cfg, ds, os.path.join(P["ck"], "long"), grid(cfg, "long_budget"))
    else:
        rows, trim_rows, trims, truth = [], [], None, None
        if args.stage == "hover":
            trims = load_trims(cfg, os.path.join(ROOT, *cfg["paths"]["results"].split("/")))
            truth = phover.trim_truth(plant, trims)
        for name, spec, seed in grid(cfg):
            p = load_params(os.path.join(P["ck"], f"{name}.npz"))
            if args.stage == "analyse":
                r = analyse_one(cfg, spec, p, ds, seed)
            else:
                r, tr_ = hover_one(cfg, spec, p, ds, plant, hx, hu, trims, truth)
                trim_rows += [dict(name=name, **t) for t in tr_]
            rows.append(dict(name=name, arch=spec.arch, init=spec.init, n_blocks=spec.n_blocks, seed=seed, **r))
        if trim_rows:
            write_csv(os.path.join(res, "hover_trims.csv"), trim_rows)
    extra = dict(p3_spearman=p3_spearman(rows)) if args.stage == "analyse" else {}
    write_csv(os.path.join(res, f"{args.stage}.csv"), rows)
    write_json(os.path.join(res, f"meta_{args.stage}.json"), meta(cfg, args.stage, extra))
    pv.print_commit_listing([os.path.join(res, f"{args.stage}.csv"), os.path.join(res, f"meta_{args.stage}.json")])


if __name__ == "__main__":
    main()
