"""P-I pre-flight (author's item P1, 5 Oct 2026): the complete pipeline end to end on a
SYNTHETIC linear plant with the real configuration, before the tag, so that the analysis
never needs changing after real data exist.

    python experiments/p1_quadrotor/preflight.py all     # every stage, then the checks
    python experiments/p1_quadrotor/preflight.py stops   # exercise the stopping rules

- Configuration: config.yaml with preflight_override.yaml on top (budgets cut: grid 2,000
  steps, long runs 4,000 with snapshots every 1,000; outputs under results/p1_preflight/).
  The override is not part of the pre-registered configuration.
- Plant: ẋ = A (x − x0) + B (u − u0), A = 0.3 N(0, 1) − I, B = N(0, 1) (seed in the
  override), Euler steps of dt, so the y-map is exactly (A, B); the quadrotor's
  dimensions, trim (x0 = 0, u0 = m g/4) and sampling box. No quadrotor dynamics, no
  quadrotor data, no surrogate of the quadrotor.
- Stages: generate; train (40) and train_long (10) through launch.py (parallel, pinned
  environment); gather; analyse; analyse_long; hover (with the committed trims); p5 for
  both stages; predictions (every rule, the race, G2). Then `check_outputs` verifies that
  every quantity prereg/p1.md promises is present (and finite where it must be).
- The gate is not bypassed: these functions are run.py's, called on synthetic data;
  run.py's own command line still refuses every quadrotor stage without the tag.
"""

import argparse
import copy
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "experiments", "r6_tdmpc2")]

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import launch  # noqa: E402
import p1_data as pdata  # noqa: E402
import p1_rules as rules  # noqa: E402
import provenance as pv  # noqa: E402
import p1_load  # noqa: E402
prun = p1_load.run()
from lens import analysis as an  # noqa: E402

OVERRIDE = os.path.join(HERE, "preflight_override.yaml")


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_cfg(config, override=OVERRIDE):
    with open(config, encoding="utf-8") as f1, open(override, encoding="utf-8") as f2:
        return merge(yaml.safe_load(f1), yaml.safe_load(f2))


def synthetic_plant(cfg):
    """The pre-flight plant (module docstring) and its sampling half-widths."""
    q, _ = pdata.quadrotor_plant(cfg["plant"])  # only its trim, bounds and dt; never stepped
    rng = np.random.default_rng(int(cfg["synthetic_plant"]["seed"]))
    A = 0.3 * rng.standard_normal((12, 12)) - np.eye(12)
    B = rng.standard_normal((12, 4))
    x0, u0, dt = np.asarray(q.x0), np.asarray(q.u0), q.dt
    f = lambda x, u: jnp.asarray(A) @ (x - x0) + jnp.asarray(B) @ (u - u0)  # noqa: E731
    plant = pdata.Plant(step=lambda x, u: x + dt * f(x, u), x0=x0, u0=u0, u_lo=q.u_lo, u_hi=q.u_hi, dt=dt, f=f)
    hx, hu = pdata.quadrotor_half_widths(cfg["sampling"]["half_widths"], q.u0)
    return plant, hx, hu


def train_index(cfg, stage, index):
    P = prun.paths(cfg)
    ds = pdata.load_dataset(P["data"], P["manifest"])
    long = stage == "train_long"
    members = prun.grid(cfg, prun.GRIDS[stage])
    ck = os.path.join(P["ck"], "long") if long else P["ck"]
    prun.stage_train(cfg, ds, ck, os.path.join(P["res"], stage), [members[index]], long, P["base"], summary=False)


def run_all(cfg, config, n_proc):
    P = prun.paths(cfg)
    res = P["res"]
    plant, hx, hu = synthetic_plant(cfg)
    if not os.path.exists(P["manifest"]):
        prun.stage_generate(cfg, plant, hx, hu, P["data"], P["manifest"])
    ds = pdata.load_dataset(P["data"], P["manifest"])
    for stage in ("train", "train_long"):
        members = prun.grid(cfg, prun.GRIDS[stage])
        cmds = [[sys.executable, __file__, "--config", config, stage, "--index", str(i)] for i in range(len(members))]
        logs = [os.path.join(res, stage, "logs", f"{n}.log") for n, _, _ in members]
        codes = launch.run_pool(cmds, n_proc, logs)
        if any(codes):
            sys.exit(f"pre-flight: {stage} runs failed: {[members[i][0] for i, c in enumerate(codes) if c]}")
        prun.stage_gather(os.path.join(res, stage), P["base"], members)
    summ = {r["name"]: r for r in prun.read_csv(os.path.join(res, "train", "train_summary.csv"))}
    rows, arrays = prun.stage_analyse(cfg, ds, P["ck"], summ, prun.grid(cfg))
    prun.write_csv(os.path.join(res, "analyse.csv"), rows)
    prun.save_npz(os.path.join(res, "analyse_lens.npz"), arrays)
    prun.write_csv(os.path.join(res, "analyse_long.csv"),
                   prun.stage_analyse_long(cfg, ds, os.path.join(P["ck"], "long"), prun.grid(cfg, "long_budget")))
    trims = prun.load_trims(cfg, os.path.join(ROOT, "results", "p1"))
    hrows, trows = prun.stage_hover(cfg, ds, P["ck"], summ, prun.grid(cfg), plant, hx, hu, trims)
    prun.write_csv(os.path.join(res, "hover.csv"), hrows)
    prun.write_csv(os.path.join(res, "hover_trims.csv"), trows)
    for which in ("train", "train_long"):
        runs, cells = prun.stage_p5(cfg, res, which)
        prun.write_csv(os.path.join(res, f"p5_{which}.csv"), runs)
        prun.write_csv(os.path.join(res, f"p5_{which}_cells.csv"), cells)
    out = prun.stage_predictions(cfg, res)
    prun.write_json(os.path.join(res, "predictions.json"), prun.meta(cfg, "preflight", dict(results=out)))
    report = check_outputs(cfg, res)
    prun.write_json(os.path.join(res, "preflight_check.json"), report)
    print(json.dumps(report["summary"], indent=1))
    return report


# --------------------------------------------------------------------------- #
# P2: every quantity prereg/p1.md promises is present                           #
# --------------------------------------------------------------------------- #

REQUIRED = {
    "train/train_summary.csv": ["best_step", "best_val", "stopped_step", "stopped_early", "diverged", "seconds"],
    "train_long/train_summary.csv": ["best_step", "best_val", "stopped_step", "diverged"],
    "analyse.csv": ["z_star_to_mean", "z_star_to_hover", "nf_ratio", "nf_near", "nf_far", "nf_n_near", "nf_n_far",
                    "affected_frac", "err_median", "err_q5", "err_q95", "kappa", "norm_c_perp", "norm_z_star",
                    "u_min_sharpness", "u_min_coverage", "u_min_D", "u_min_r_eff", "d1_sharpness", "d1_coverage",
                    "r_star_init_median", "rel_mse_val", "rel_mse_test", "dist_kind"],
    "analyse_long.csv": ["step", "nf_ratio", "u_min_r_eff_over_D", "z_star_to_mean", "kappa"],
    "hover.csv": ["rel_err_A", "rel_err_B", "sign_agree_A", "sign_agree_B", "n_sign_A", "n_sign_B",
                  "n_sign_disagree_unmasked_A", "n_sign_disagree_unmasked_B", "true_closed_loop_abscissa",
                  "no_stabilising_gain", "stable", "h1_fail", "trims_variation_err_median",
                  "trims_spearman_dist_variation_err", "trims_rel_err_A_median", "trims_spearman_dist_rel_err_A"],
    "hover_trims.csv": ["trim", "lens_distance", "rel_err_A", "rel_err_B", "sign_agree_A", "sign_agree_B",
                        "variation_err", "true_variation"],
    "p5_train.csv": ["end_step", "ratio_init", "ratio_end", "fold_end", "kappa_end", "ratio_max", "passes"],
    "p5_train_long.csv": ["end_step", "ratio_init", "ratio_end", "fold_end", "kappa_end", "ratio_max", "passes"],
    "p5_train_cells.csv": ["arch", "n_blocks", "n", "n_pass", "holds"],
    "p5_train_long_cells.csv": ["arch", "n", "n_pass", "holds"],
}
HISTORY = ["step", "train_mse", "val_mse", "kappa", "norm_c_perp", "norm_z_star", "S_u_min", "S_d1", "D_u_min",
           "D_d1", "r_star_u_min", "r_eff_u_min", "r_eff_over_D_u_min", "r_eff_over_D_d1"]
STACK = ["p4_P_ratio_u_min", "p4_P_out_u_min", "p4_P_1_u_min", "p4_S_ratio_u_min", "p4_S_out_u_min",
         "p4_S_block1_u_min"]
PRED = ["p1", "p1_overall", "g2", "p2", "p3", "p4", "p5", "h1", "race"]


def check_outputs(cfg, res):
    """Every table and quantity p1.md lists, present and (where it must be) finite."""
    problems, files = [], {}
    for rel, cols in REQUIRED.items():
        rows = prun.read_csv(os.path.join(res, *rel.split("/")))
        files[rel] = len(rows)
        for r in rows:
            if rules.diverged(r):
                continue
            miss = [c for c in cols if c not in r or r[c] == ""]
            if miss:
                problems.append(f"{rel}: {r.get('name', '?')} lacks {miss}")
                break
    an_rows = prun.read_csv(os.path.join(res, "analyse.csv"))
    for r in an_rows:
        if int(r["n_blocks"]) > 1 and not rules.diverged(r):
            miss = [c for c in STACK if r.get(c, "") == ""]
            if miss:
                problems.append(f"analyse.csv: {r['name']} lacks {miss}")
        if not rules.diverged(r) and r["degenerate"] == "False" and r.get("corollary1_max_rel_dev", "") == "":
            problems.append(f"analyse.csv: {r['name']} lacks the Corollary 1 deviation")
    for stage in ("train", "train_long"):
        for name, _, _ in prun.grid(cfg, prun.GRIDS[stage]):
            h = prun.read_csv(os.path.join(res, stage, "history", f"{name}.csv"))
            miss = [c for c in HISTORY if c not in h[0]]
            if miss:
                problems.append(f"{stage}/history/{name}.csv lacks {miss}")
            a = np.load(os.path.join(res, stage, "history", f"{name}.npz"))
            if not {"step", "z_star", "widths", "widths_eff", "u_min"} <= set(a.files):
                problems.append(f"{stage}/history/{name}.npz lacks arrays")
    lens = np.load(os.path.join(res, "analyse_lens.npz"))
    files["analyse_lens.npz"] = len(lens.files)
    with open(os.path.join(res, "predictions.json"), encoding="utf-8") as f:
        pred = json.load(f)["results"]
    problems += [f"predictions.json lacks {k}" for k in PRED if k not in pred]
    for stage in ("train", "train_long"):
        for fn in ("train_summary.csv", "outputs_manifest.csv"):
            if not os.path.exists(os.path.join(res, stage, fn)):
                problems.append(f"{stage}/{fn} missing")
    long_rows = prun.read_csv(os.path.join(res, "analyse_long.csv"))
    n_snap = len({(r["name"], r["step"]) for r in long_rows})
    want = len(prun.grid(cfg, "long_budget")) * (int(cfg["long_budget"]["max_steps"])
                                                 // int(cfg["long_budget"]["snapshot_every"]) + 1)
    if n_snap != want:
        problems.append(f"analyse_long.csv has {n_snap} snapshot rows, expected {want}")
    summary = dict(
        problems=problems, files=files,
        p1_cells=[(c["arch"], c["n_blocks"], c["n_pass"], c["holds"]) for c in pred["p1"]],
        p1_overall=pred["p1_overall"], g2=pred["g2"],
        p2=dict(a_holds=pred["p2"]["a_holds"], holds=pred["p2"]["holds"]),
        p3=dict(rho=pred["p3"]["rho"], holds=pred["p3"]["holds"],
                within={k: v["rho"] for k, v in pred["p3"]["within_init"].items()}),
        p4=dict(holds=pred["p4"]["holds"], median=pred["p4"]["zero_bias_median"],
                consistent=pred["p4"]["point_consistent"]),
        p5=dict(holds=pred["p5"]["holds"]), h1=dict(holds=pred["h1"]["holds"]), race=pred["race"])
    return dict(summary=summary, predictions=pred)


# --------------------------------------------------------------------------- #
# P4: the stopping rules, exercised on purpose                                  #
# --------------------------------------------------------------------------- #


def exercise_stops(cfg, scratch):
    """Each injected fault must stop with a clear message; returns {fault: message}."""
    P = prun.paths(cfg)
    res = P["res"]
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch)
    out = {}
    # 1. a data-manifest mismatch
    d = os.path.join(scratch, "data")
    shutil.copytree(P["data"], d)
    with open(os.path.join(d, "train.npz"), "ab") as f:
        f.write(b"\0")
    try:
        pdata.load_dataset(d, P["manifest"])
        out["manifest_mismatch"] = "NOT STOPPED"
    except pv.ProvenanceError as e:
        out["manifest_mismatch"] = str(e)
    # 2. an injected NaN: the run stops, is recorded as diverged and fails every rule
    ds = pdata.load_dataset(P["data"], P["manifest"])
    bad = copy.deepcopy(ds)
    bad["train"]["Y"] = bad["train"]["Y"].copy()
    bad["train"]["Y"][0, 0] = np.nan
    name, spec, seed = prun.grid(cfg)[0]
    row = prun.stage_train(cfg, bad, os.path.join(scratch, "ck"), os.path.join(scratch, "train"), [(name, spec, seed)],
                           base=scratch, summary=True)[0]
    out["injected_nan"] = (f"{name}: diverged = {row['diverged']} at step {row['stopped_step']}"
                           if row["diverged"] else "NOT STOPPED")
    # 3. a dirty worktree at gather time (the run JSON records git_dirty)
    rd = os.path.join(scratch, "dirty")
    shutil.copytree(os.path.join(scratch, "train"), rd)
    rj = os.path.join(rd, "runs", f"{name}.json")
    m = json.load(open(rj, encoding="utf-8"))
    m["git_dirty"] = True
    prun.write_json(rj, m)
    files = []
    for f in pv.read_manifest(os.path.join(rd, "runs", f"{name}_manifest.csv")):
        parts = f["file"].replace("\\", "/").split("/")
        files.append(os.path.join(scratch, *(["dirty"] + parts[1:] if parts[0] == "train" else parts)))
    prun.write_manifest(os.path.join(rd, "runs", f"{name}_manifest.csv"), files, scratch)
    try:
        prun.stage_gather(rd, scratch, [(name, spec, seed)])
        out["dirty_worktree_at_gather"] = "NOT STOPPED"
    except pv.ProvenanceError as e:
        out["dirty_worktree_at_gather"] = str(e)
    # 4. a missing run
    try:
        prun.stage_gather(os.path.join(res, "train"), P["base"], prun.grid(cfg) + [("missing-run", spec, 99)])
        out["missing_run"] = "NOT STOPPED"
    except pv.ProvenanceError as e:
        out["missing_run"] = str(e)
    # 5. a forced Corollary 1 deviation
    summ = {r["name"]: r for r in prun.read_csv(os.path.join(res, "train", "train_summary.csv"))}
    real = an.corollary1_line
    an.corollary1_line = lambda *a, **k: dict(real(*a, **k), max_rel_dev=1e-6)
    try:
        prun.stage_analyse(cfg, ds, P["ck"], summ, prun.grid(cfg)[:3])
        out["corollary1_deviation"] = "NOT STOPPED"
    except RuntimeError as e:
        out["corollary1_deviation"] = str(e)
    finally:
        an.corollary1_line = real
    prun.write_json(os.path.join(res, "preflight_stops.json"), out)
    for k, v in out.items():
        print(f"{k}: {v}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("stage", choices=["all", "train", "train_long", "stops"])
    ap.add_argument("--index", type=int)
    ap.add_argument("--n-proc", type=int, default=launch.N_PROC)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.stage in ("train", "train_long"):
        train_index(cfg, args.stage, args.index)
    elif args.stage == "all":
        run_all(cfg, args.config, args.n_proc)
    else:
        exercise_stops(cfg, os.path.join(prun.paths(cfg)["res"], "stops_scratch"))


if __name__ == "__main__":
    main()
