"""R6 criterion stage: r6.md's criterion and G1 with D3-D7, on the D6 planner data.

    python experiments/r6_tdmpc2/criterion.py --config experiments/r6_tdmpc2/config.yaml \
        --drive "G:\\My Drive\\layernorm-lens-r6"

Runs on the author's laptop in float64 (D7 (8)); JAX float64 for D6 (g), torch only
to read the .pt checkpoints. Before computing anything it refuses to run unless:

  - the five annotated tags prereg-r6, -d1, -d2, -d3, -d4 exist locally, the five
    pre-registration files match them, and the same tag objects are on origin;
  - the working tree has no uncommitted changes to tracked files;
  - every checkpoint matches D6's SHA-256 table, read from the tag prereg-r6-d3;
  - every planner observation file (kind "data" rows of
    results/r6/planner/planner_obs_manifest.csv; smoke files are never read) and every
    D4 observation file (results/r6/d4_obs_manifest.csv, with match: true in
    meta_d4_regeneration.json) is on Drive with its SHA-256, and in D7 (3)'s order;
  - each planner meta records a complete 50-episode eval_mode=True data run of that
    checkpoint with the manifest's SHA-256 and the input D6 (b) requires; for the two
    pre-release checkpoints, controls as stated and a passed encoder gate; its
    returns agree with the stored rewards;
  - each checkpoint's lens, recomputed from its weights, equals the lens stage's
    results/r6/lens/<task>-seed<seed>.npz, and the JAX networks equal layouts.py's.

Then, in order, stopping at any stop rule (run_stages):
  1. the Corollary 1 checks, D7 (4) and D6 (g) lines; stop on any > 1e-10;
  2. D6 (b) condition (ii) for cartpole-swingup seed 1 on the planner data (D7 (2)),
     the record-only value on the D4 data, the humanoid-run seed 3 label, the D6 (c)
     calibration;
  3. Inside, Populated, Sharp for all 15 checkpoints, under all three readings for
     the two pre-release checkpoints (D7 (5));
  4. the D7 (1) tie rule, the counting checkpoints and the G1 vote;
  5. D6 (g) susceptibility, reported only;
  6. the D7 (8) borderline scan: if anything is borderline, the results are written
     but the stage is marked "pending Colab rerun".

Outputs in results/r6/criterion/ (or --out), copied to Drive under the same relative
path and verified there; the run ends by printing the exact `git add -f` command.
"""

import argparse
import csv
import dataclasses
import datetime
import glob
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import collect  # noqa: E402  (D3's computation: consistency_errors)
import criterion_lib as cl  # noqa: E402
import layouts  # noqa: E402
import provenance as pv  # noqa: E402

geo = cl.geo

STATUS_COMPLETE = "complete"
STATUS_PENDING = "pending Colab rerun (D7 (8))"
STATUS_STOPPED = "stopped: Corollary 1 check above tolerance (D7 (4))"

OUTPUTS = """
corollary1.csv             D7 (4) and D6 (g): every line, its own s*, r*, kappa, T, max relative deviation
criterion.csv              one row per checkpoint and reading: Inside, Populated, Sharp, criterion
rho_fractions.csv          r6.md, reported regardless: rho_eff and rho fractions (primary reading)
consistency_planner.csv    D3 on the planner data (pre-release and D6 (c) calibration)
consistency_d4.csv         D3 on the D4 data for the pre-release checkpoints (record only)
identification.json        D6 (b) with D7 (1), (2): cartpole-swingup seed 1; humanoid-run seed 3 label
g1.json, g1.csv            G1: counting checkpoint per task, its result, the vote
susceptibility.csv         D6 (g): per checkpoint, direction and output
borderline.json            D7 (8): every scanned statistic, its threshold, the 1% flag; the stage status
lines/<name>.npz           D6 (g): t grid, |J(t)| and |g'| along each line (float64)
populated/<name>-<reading>.npz  Populated: subsample indices and nearest-other distances
meta_criterion.json        status, config, commit, tags, versions, every input's SHA-256, tdmpc2 init.py
"""


def die(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def name_of(task, seed):
    return f"{task}-seed{seed}"


# --------------------------------------------------------------------------- #
# one checkpoint in memory                                                      #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Checkpoint:
    task: str
    seed: int
    layout: str
    sd: dict                 # float64 numpy state_dict (encoder, dynamics, policy)
    mods: list               # module names from the checkpoint's _metadata, or None
    O: np.ndarray            # (episodes, steps + 1, k) raw observations, float64
    A: np.ndarray            # (episodes, steps, a) executed actions, float64
    fraction: float          # planner return / published (D6 (a), (b) condition (i))
    flag: bool               # below 0.5 x published (D6 (a))

    @property
    def name(self):
        return name_of(self.task, self.seed)


def primary_reading(layout):
    """D6 (b): symlog at position 0 for the pre-release checkpoints; the public
    layout's layer receives the flattened observation (D5)."""
    return "symlog" if layout == layouts.PRERELEASE else "identity"


def readings_of(cfg, layout):
    """D7 (5): all three D3 readings for the two pre-release checkpoints, the primary
    one first; identity alone for the public layout."""
    if layout != layouts.PRERELEASE:
        return ["identity"]
    cands = list(cfg["consistency"]["candidates"])
    return ["symlog"] + [c for c in cands if c != "symlog"]


def networks(cfg, sd, mods, layout, cand):
    """numpy networks as D3 builds them (collect.agent), with D3 candidate `cand` at
    the encoder's input."""
    c = cfg["consistency"]
    eps, sdim = float(cfg["layer"]["layernorm_eps"]), int(c["simnorm_dim"])
    inp = layouts.input_candidates(float(c["pos0_layernorm_eps"]))[cand]
    enc, dyn, pi = layouts.build_networks(sd, layout, eps, sdim, mods, input_fn=inp)
    return dict(enc=enc, dyn=dyn, pi=pi)


@dataclasses.dataclass
class Prepared:
    ck: Checkpoint
    E: np.ndarray
    b: np.ndarray
    L: object
    X: dict                  # reading -> (n, k) layer-input coordinates, float64
    S: dict                  # reading -> cl.DataStats
    primary: str
    enc_j: object
    dyn_j: object


def prepare(cfg, ck):
    """Layer-1 weights and lens, the data in every reading, and the JAX networks
    checked against layouts.py's (a pipeline check, before any rule)."""
    cc = cfg["criterion"]
    eps, sdim = float(cfg["layer"]["layernorm_eps"]), int(cfg["consistency"]["simnorm_dim"])
    lin, _ = layouts.check_first_layer(ck.sd, ck.layout, cfg["layer"]["layouts"], ck.name)
    E, b = ck.sd[f"{lin}.weight"], ck.sd[f"{lin}.bias"]
    L = geo.lens(E, b, eps)
    fns = layouts.input_candidates(float(cfg["consistency"]["pos0_layernorm_eps"]))
    flat = np.asarray(ck.O, np.float64).reshape(-1, ck.O.shape[-1])
    X = {r: fns[r](flat) for r in readings_of(cfg, ck.layout)}
    S = {r: cl.data_stats(x, float(cc["kept_rtol"])) for r, x in X.items()}
    primary = primary_reading(ck.layout)

    num = networks(cfg, ck.sd, ck.mods, ck.layout, "identity")  # layer-input coordinates
    enc_j = cl.jax_mlp(cl.mlp_spec(ck.sd, ck.layout, "_encoder.state", "simnorm"), eps, sdim)
    dyn_j = cl.jax_mlp(cl.mlp_spec(ck.sd, ck.layout, "_dynamics", "simnorm"), eps, sdim)
    Xp = X[primary]
    probe = Xp[np.random.default_rng(0).choice(len(Xp), min(256, len(Xp)), replace=False)]
    zp = np.concatenate([num["enc"](probe), np.zeros((len(probe), ck.A.shape[-1]))], -1)
    for what, f_np, f_j, x in (("encoder", num["enc"], enc_j, probe), ("dynamics", num["dyn"], dyn_j, zp)):
        dev = float(np.max(np.abs(f_np(x) - np.asarray(f_j(jnp.asarray(x))))))
        if dev > 1e-12:
            die(f"{ck.name}: JAX {what} differs from layouts.py by {dev:.3g}.")
    return Prepared(ck=ck, E=E, b=b, L=L, X=X, S=S, primary=primary, enc_j=enc_j, dyn_j=dyn_j)


def g_directions(P):
    """D6 (g) Directions: d₁, d₂ (None if degenerate) and u_min, signed by D7 (7)."""
    S, L = P.S[P.primary], P.L
    return dict(d1=S.d1, d2=S.d2 if S.d2_kept else None,
                u_min=cl.canonical_sign(L.principal_dirs[:, int(np.argmin(L.principal_widths))]))


# --------------------------------------------------------------------------- #
# the six steps                                                                 #
# --------------------------------------------------------------------------- #


def step1_corollary1(cfg, preps):
    """D7 (4) lines through μ (d₁, d₂) and through 10 data states (d₁); D6 (g) lines
    through z* (d₁, d₂, u_min). All in the primary coordinates. Returns the rows, the
    D6 (g) grids for step 5, and whether every line is within tolerance."""
    cc = cfg["criterion"]
    c1, sus = cc["corollary1"], cc["susceptibility"]
    n_grid, t_range, rtol = int(sus["n_grid"]), float(sus["t_range_r_eff"]), float(c1["rtol"])
    rows, grids = [], {}
    for key, P in preps.items():
        S, X, L = P.S[P.primary], P.X[P.primary], P.L
        base = dict(task=P.ck.task, seed=P.ck.seed, reading=P.primary)
        if L.degenerate:  # no checkpoint is degenerate (lens_summary.csv); kept for completeness
            rows.append(dict(base, line="all", status="degenerate lens: Corollary 1 not applicable"))
            continue
        lines = [("D7 (4)", "mu + s d1", S.mu, S.d1)]
        if S.d2_kept:
            lines.append(("D7 (4)", "mu + s d2", S.mu, S.d2))
        else:
            rows.append(dict(base, rule="D7 (4)", line="mu + s d2", status="d2 degenerate (D6 (g)): skipped"))
        for i in cl.data_line_rows(S.n, int(c1["n_state_lines"]), int(c1["rng_seed"])):
            lines.append(("D7 (4)", f"x[{int(i)}] + s d1", X[int(i)], S.d1))
        for rule, label, x0, d in lines:
            r = cl.corollary1_data_line(L, P.E, P.b, x0, d, X, n_grid, t_range)
            rows.append(dict(base, rule=rule, line=label, status="ok" if r["max_rel_dev"] <= rtol else "FAIL",
                             **r))
        for dname, d in g_directions(P).items():
            if d is None:
                rows.append(dict(base, rule="D6 (g)", line=f"z* + t {dname}", status="d2 degenerate (D6 (g)): skipped"))
                continue
            t, info, dev = cl.corollary1_g_line(L, P.E, P.b, d, X, n_grid, t_range)
            grids[(key, dname)] = (t, info)
            rows.append(dict(base, rule="D6 (g)", line=f"z* + t {dname}", status="ok" if dev <= rtol else "FAIL",
                             s_star=0.0, r_star_l=info["r_star"], r_eff_l=info["r_eff"], kappa_l=float(L.kappa),
                             T=info["T"], n_grid=n_grid, max_rel_dev=dev))
    ok = all(r["status"] != "FAIL" for r in rows)
    return rows, grids, ok


def step2_identification(cfg, preps, d4=None):
    """D3's computation on the planner data for the pre-release checkpoints and the
    D6 (c) calibration checkpoints; on the D4 data (record only, D6 (b)); condition
    (ii) with D7 (2), the D7 (1) ratios, condition (i), the humanoid-run seed 3 label.
    d4: {(task, seed): (O, A)} of D4 observations for the pre-release checkpoints."""
    c, cc = cfg["consistency"], cfg["criterion"]
    idc = cc["identification"]
    gate_k = (idc["gate"]["task"], int(idc["gate"]["seed"]))
    rep_k = (idc["reported"]["task"], int(idc["reported"]["seed"]))
    jobs = [(gate_k, "gate (D6 (b))"), (rep_k, "reported (D6 (b))")] + \
           [((t["task"], int(t["seed"])), "calibration (D6 (c))") for t in cc["calibration"]]
    err_planner, cons_rows = {}, []
    for k, role in jobs:
        ck = preps[k].ck
        ags = {cand: networks(cfg, ck.sd, ck.mods, ck.layout, cand) for cand in c["candidates"]}
        err = collect.consistency_errors(ags, ck.O, ck.A, int(c["rng_seed"]))
        err_planner[k] = err
        for cand, x in err.items():
            cons_rows.append(dict(task=k[0], seed=k[1], role=role, layout=ck.layout, data="planner",
                                  candidate=cand, e=x["e"], e0=x["e0"], e_over_e0=x["e"] / x["e0"], n=x["n"]))
    err_d4, d4_rows = {}, []
    for k, (O, A) in (d4 or {}).items():
        ck = preps[k].ck
        ags = {cand: networks(cfg, ck.sd, ck.mods, ck.layout, cand) for cand in c["candidates"]}
        err = collect.consistency_errors(ags, O, A, int(c["rng_seed"]))
        err_d4[k] = err
        for cand, x in err.items():
            d4_rows.append(dict(task=k[0], seed=k[1], data="D4 policy prior (record only)", candidate=cand,
                                e=x["e"], e0=x["e0"], e_over_e0=x["e"] / x["e0"], n=x["n"]))
    cand, emax = idc["candidate"], float(idc["e_over_e0_max"])
    cii_gate = cl.condition_ii(err_planner[gate_k], cand, emax)
    cii_rep = cl.condition_ii(err_planner[rep_k], cand, emax)
    tie = cl.tie_rule(err_planner[gate_k], cand, float(cc["tie_ratio"]))
    gate, rep = cl.identification(preps[gate_k].ck.fraction, cii_gate, preps[rep_k].ck.fraction, cii_rep,
                                  float(idc["return_ratio"]))
    h1 = err_planner.get(("humanoid-run", 1))

    def ratios(err):
        return {c: v["e"] / v["e0"] for c, v in err.items()}

    ident = {
        name_of(*gate_k): dict(**gate, condition_ii_detail=cii_gate, tie_rule=tie,
                               e_over_e0_planner=ratios(err_planner[gate_k]),
                               condition_ii_on_d4_record_only=(cl.condition_ii(err_d4[gate_k], cand, emax)
                                                               if gate_k in err_d4 else None)),
        name_of(*rep_k): dict(**rep, e_over_e0_planner=ratios(err_planner[rep_k]),
                              symlog_lowest_on_d4_record_only=(cl.condition_ii(err_d4[rep_k], cand, emax)["lowest"]
                                                               if rep_k in err_d4 else None),
                              humanoid_run_seed1_calibration_e_over_e0=ratios(h1) if h1 else None),
    }
    return dict(cons_rows=cons_rows, d4_rows=d4_rows, err_planner=err_planner, err_d4=err_d4,
                cii_gate=cii_gate, tie=tie, gate=gate, rep=rep, ident=ident, gate_key=gate_k)


def step3_criterion(cfg, preps):
    """r6.md's criterion for every checkpoint; every reading for the pre-release ones
    (D7 (5)); ρ fractions (reported regardless) in the primary reading."""
    cc = cfg["criterion"]
    crit_rows, rho_rows, pop = [], [], {}
    for key, P in preps.items():
        ck, L = P.ck, P.L
        for reading in readings_of(cfg, ck.layout):
            S, X = P.S[reading], P.X[reading]
            crit, idx, nn = cl.criterion(S, X, L, cc)
            crit_rows.append(dict(task=ck.task, seed=ck.seed, layout=ck.layout, reading=reading,
                                  primary=reading == P.primary, n=S.n, k=S.k, m=S.m,
                                  E_shape=f"{P.E.shape[0]}x{P.E.shape[1]}", eps=float(L.eps),
                                  norm_z_star=float(np.linalg.norm(L.z_star)), kappa=float(L.kappa),
                                  degenerate=bool(L.degenerate), lambda1=float(S.eigvals[0]),
                                  sharp_min=float(cc["sharp_min"]), fraction=ck.fraction,
                                  flag_below_half=ck.flag, **crit))
            pop[(key, reading)] = (idx, nn)
        rho_rows.append(dict(task=ck.task, seed=ck.seed, reading=P.primary,
                             **cl.rho_fractions(L, P.X[P.primary], cc["rho_levels"])))
    return crit_rows, rho_rows, pop


def step4_g1(cfg, crit_rows, s2):
    """D7 (1) for cartpole-swingup seed 1, the counting checkpoints (r6.md, D6 (a),
    (b)) and the G1 vote."""
    cc = cfg["criterion"]
    gate_k = s2["gate_key"]
    primary = {(r["task"], r["seed"]): r["criterion"] for r in crit_rows if r["primary"]}
    flags = {(r["task"], r["seed"]): r["flag_below_half"] for r in crit_rows if r["primary"]}
    by_reading = {r["reading"]: r["criterion"] for r in crit_rows if (r["task"], r["seed"]) == gate_k}
    cart = cl.cartpole_seed1(s2["gate"]["identified"], s2["tie"], by_reading, cc["identification"]["candidate"])
    rows = cl.counting_checkpoints(cc["g1"]["tasks"], primary, cart, flags, cartpole_task=gate_k[0])
    g1 = cl.g1_vote(rows, int(cc["g1"]["min_pass"]))
    return dict(cartpole_seed1=cart, **g1)


def step5_susceptibility(cfg, preps, grids):
    """D6 (g), reported only: encoder output (after SimNorm) and the dynamics
    prediction with a = 0, along d₁, d₂, u_min through z*, in the primary coordinates."""
    t_range = float(cfg["criterion"]["susceptibility"]["t_range_r_eff"])
    sus_rows, lines = [], {}
    for key, P in preps.items():
        ck, L, S, X = P.ck, P.L, P.S[P.primary], P.X[P.primary]
        a_dim, k = ck.A.shape[-1], S.k
        enc_j, dyn_j = P.enc_j, P.dyn_j

        def enc_f(x, enc_j=enc_j):
            return enc_j(x)

        def dyn_a0(x, enc_j=enc_j, dyn_j=dyn_j, a_dim=a_dim):
            return dyn_j(jnp.concatenate([enc_j(x), jnp.zeros(a_dim)]))

        def dyn_rec(xa, enc_j=enc_j, dyn_j=dyn_j, k=k):
            return dyn_j(jnp.concatenate([enc_j(xa[:k]), xa[k:]]))

        # D7 (6): observation t paired with the action executed from it, t = 0..499
        Xa = np.concatenate([X.reshape(ck.O.shape[0], ck.O.shape[1], -1)[:, :-1].reshape(-1, k),
                             np.asarray(ck.A, np.float64).reshape(-1, a_dim)], -1)
        arrays = dict(z_star=L.z_star)
        for dname, d in g_directions(P).items():
            arrays[dname] = d if d is not None else np.full(k, np.nan)
            if d is None or (key, dname) not in grids:
                sus_rows.append(dict(task=ck.task, seed=ck.seed, direction=dname,
                                     status="degenerate (D6 (g))" if d is None else "degenerate lens"))
                continue
            t, info = grids[(key, dname)]
            dpad = np.concatenate([d, np.zeros(a_dim)])
            outs = {"encoder": (enc_f, {"a0": (enc_f, X, d)}),
                    "dynamics": (dyn_a0, {"a0": (dyn_a0, X, d), "recorded": (dyn_rec, Xa, dpad)})}
            res = cl.susceptibility(L, d, t, info, outs, t_range)
            arrays[f"{dname}_t"] = t
            for oname, o in res.items():
                arrays[f"{dname}_{oname}_J"] = o.pop("J_line")
                arrays[f"{dname}_{oname}_g_prime"] = o.pop("g_prime_line")
                sus_rows.append(dict(task=ck.task, seed=ck.seed, reading=P.primary, direction=dname,
                                     output=oname, status="ok", r_star=info["r_star"], r_eff=info["r_eff"],
                                     T=info["T"], fraction=ck.fraction, flag_below_half=ck.flag, **o))
        lines[ck.name] = arrays
    return sus_rows, lines


def step6_borderline(cfg, crit_rows, s2):
    items = cl.borderline_items(crit_rows, s2["cii_gate"], s2["tie"], name_of(*s2["gate_key"]))
    scan = cl.borderline_scan(items, float(cfg["criterion"]["borderline_band"]))
    return scan, any(s["borderline"] for s in scan)


def run_stages(cfg, cks, d4=None, log=print):
    """The six steps in order, stopping at any stop rule. cks: {(task, seed):
    Checkpoint}, already verified. Returns everything to write, with 'status'."""
    preps = {k: prepare(cfg, ck) for k, ck in cks.items()}
    res = dict(status=None, preps=preps)

    log("step 1: Corollary 1 checks (D7 (4), D6 (g))")
    res["cor1_rows"], grids, ok = step1_corollary1(cfg, preps)
    worst = max((r["max_rel_dev"] for r in res["cor1_rows"] if "max_rel_dev" in r), default=float("nan"))
    log(f"  {len(res['cor1_rows'])} lines, max relative deviation {worst:.3g}")
    if not ok:
        res["status"] = STATUS_STOPPED
        return res

    log("step 2: identification (D6 (b), D7 (1), (2)) and calibration (D6 (c))")
    res["s2"] = step2_identification(cfg, preps, d4)
    log("step 3: Inside, Populated, Sharp (r6.md; D7 (3), (5))")
    res["crit_rows"], res["rho_rows"], res["pop"] = step3_criterion(cfg, preps)
    log("step 4: tie rule, counting checkpoints, G1 (D7 (1), D6 (a), (b), r6.md)")
    res["g1"] = step4_g1(cfg, res["crit_rows"], res["s2"])
    log("step 5: susceptibility (D6 (g), reported only)")
    res["sus_rows"], res["lines"] = step5_susceptibility(cfg, preps, grids)
    log("step 6: borderline scan (D7 (8))")
    res["borderline"], pending = step6_borderline(cfg, res["crit_rows"], res["s2"])
    res["band"] = float(cfg["criterion"]["borderline_band"])
    res["status"] = STATUS_PENDING if pending else STATUS_COMPLETE
    return res


# --------------------------------------------------------------------------- #
# inputs: provenance and verification                                           #
# --------------------------------------------------------------------------- #


def drive_path(drive, rel):
    """A Drive file from a manifest's or the config's "/"-separated relative path."""
    return os.path.join(drive, *rel.split("/"))


def verify_file(path, sha, what):
    if not os.path.exists(path):
        die(f"{what}: {path} not found.")
    try:
        pv.verify_sha256(path, sha, what)
    except pv.ProvenanceError as e:
        die(str(e))


def planner_inputs(cfg, drive, checkpoints):
    """Planner metas and the manifest's kind "data" rows: refuse anything incomplete
    or inconsistent; verify every data file on Drive (SHA-256) and its order (D7 (3))."""
    cc, pc = cfg["criterion"], cfg["planner_collect"]
    man_path = os.path.join(ROOT, cc["planner_manifest"])
    rows = [r for r in pv.read_manifest(man_path) if r["kind"] == "data"]
    by_name = {r["file"].rsplit("/", 1)[-1]: r for r in rows}
    want_names = {f"{name_of(t, s)}.npz" for t, s, _ in checkpoints}
    if set(by_name) != want_names or len(rows) != len(want_names):
        die(f"{cc['planner_manifest']}: kind 'data' rows are {sorted(by_name)}, expected {sorted(want_names)}.")
    n_ep = len(pc["env_seeds"]) * int(pc["episodes_per_env_seed"])
    out, rec = {}, dict(manifest=cc["planner_manifest"], manifest_sha256=pv.sha256(man_path), files={}, metas={})
    for task, seed, layout in checkpoints:
        n = name_of(task, seed)
        mp = os.path.join(ROOT, cc["planner_results"], f"meta_{n}.json")
        if not os.path.exists(mp):
            die(f"{mp} not found: no planner data for {n}.")
        m = json.load(open(mp))
        if not (m.get("complete") is True and not m.get("smoke") and m.get("kind") == "data"
                and m["returns"]["episodes"] == n_ep and m.get("eval_mode") is True):
            die(f"{n}: planner meta is not a complete {n_ep}-episode eval_mode=True data run.")
        if m["layout"] != layout or m["task"] != task or m["seed"] != seed:
            die(f"{n}: planner meta records {m['task']} seed {m['seed']} [{m['layout']}].")
        if m["planner_input"] != primary_reading(layout):
            die(f"{n}: planner acted on {m['planner_input']}, D6 (b) requires {primary_reading(layout)}.")
        if layout == layouts.PRERELEASE and not ((m.get("controls") or {}).get("as_stated")
                                                 and (m.get("gate_result") or {}).get("pass")):
            die(f"{n}: its meta does not record controls as stated and a passed encoder gate.")
        row = by_name[f"{n}.npz"]
        if row["sha256"] != m["data"]["sha256"]:
            die(f"{n}: manifest SHA-256 {row['sha256']} differs from the meta's {m['data']['sha256']}.")
        path = drive_path(drive, row["file"])
        verify_file(path, row["sha256"], f"planner observations {row['file']}")
        d = np.load(path)
        if not cl.episode_order_ok(d["env_seed"], d["episode_in_seed"], pc["env_seeds"],
                                   int(pc["episodes_per_env_seed"])):
            die(f"{n}: env_seed / episode_in_seed are not in D7 (3)'s order.")
        O, A, R = d["obs"], d["actions"], d["rewards"]
        if O.shape[:2] != (n_ep, int(pc["steps"]) + 1) or A.shape[:2] != (n_ep, int(pc["steps"])):
            die(f"{n}: obs {O.shape}, actions {A.shape}; expected ({n_ep}, {int(pc['steps']) + 1}, k), "
                f"({n_ep}, {pc['steps']}, a).")
        ret = np.asarray(R, np.float64).sum(axis=1).mean()
        if not np.isclose(ret, m["returns"]["return_mean"], rtol=1e-6, atol=0):
            die(f"{n}: stored rewards give mean return {ret}, the meta records {m['returns']['return_mean']}.")
        frac = float(m["returns"]["fraction"])
        flag = bool(frac < float(pc["flag_below"]))
        if flag != bool(m["returns"]["flag_below_half"]):
            die(f"{n}: the < {pc['flag_below']} flag disagrees with the meta.")
        out[(task, seed)] = (np.asarray(O, np.float64), np.asarray(A, np.float64), frac, flag)
        rec["files"][row["file"]] = row["sha256"]
        rec["metas"][os.path.relpath(mp, ROOT).replace(os.sep, "/")] = pv.sha256(mp)
    return out, rec


def d4_inputs(cfg, drive, keep):
    """D4 observations: match: true in meta_d4_regeneration.json; every file in
    d4_obs_manifest.csv on Drive with its SHA-256 and environment seeds in order.
    Returns {(task, seed): (O, A)} for `keep` (the pre-release checkpoints)."""
    cc, dc = cfg["criterion"], cfg["data"]
    meta_p = os.path.join(ROOT, cc["d4_meta"])
    if json.load(open(meta_p)).get("match") is not True:
        die(f"{cc['d4_meta']} does not record match: true.")
    man_p = os.path.join(ROOT, cc["d4_manifest"])
    rows = pv.read_manifest(man_p)
    want = {f"{name_of(t, s)}.npz" for t in cfg["tasks"] for s in cfg["seeds"]}
    if {r["file"] for r in rows} != want:
        die(f"{cc['d4_manifest']} lists {sorted(r['file'] for r in rows)}, expected {sorted(want)}.")
    per_seed = int(dc["episodes_per_task"]) // len(dc["env_seeds"])
    out, rec = {}, dict(meta=cc["d4_meta"], meta_sha256=pv.sha256(meta_p), manifest=cc["d4_manifest"],
                        manifest_sha256=pv.sha256(man_p), files={})
    for r in rows:
        path = drive_path(drive, f"{cc['d4_drive_dir']}/{r['file']}")
        verify_file(path, r["sha256"], f"D4 observations {r['file']}")
        d = np.load(path)
        if not cl.episode_order_ok(d["env_seed"], None, dc["env_seeds"], per_seed):
            die(f"D4 {r['file']}: env_seed is not in order.")
        rec["files"][r["file"]] = r["sha256"]
        for k in keep:
            if r["file"] == f"{name_of(*k)}.npz":
                out[k] = (np.asarray(d["obs"], np.float64), np.asarray(d["actions"], np.float64))
    return out, rec


def load_checkpoint(cfg, table, task, seed):
    """Download and verify (D6 table, from the tag), then a float64 numpy state_dict."""
    import torch
    name = pv.checkpoint_name(task, seed)
    repo = cfg["source"]["checkpoints"].rstrip("/").split("huggingface.co/")[1]
    url = pv.HF_FILE.format(repo=repo, rev=cfg["source"]["hf_revision"], path=name)
    local = os.path.join(ROOT, cfg["paths"]["checkpoints"], *name.split("/"))
    try:
        sha = pv.download_verified(url, local, table[name], name)
    except pv.ProvenanceError as e:
        die(str(e))
    ck = torch.load(local, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    mods = getattr(sd, "_metadata", None)
    keep = ("_encoder.", "_dynamics.", "_pi.")
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in sd.items()
          if hasattr(v, "detach") and k.startswith(keep)}
    return sd, (None if mods is None else sorted(mods.keys())), sha


def check_lens(cfg, L, task, seed, rtol=1e-12):
    """The recomputed lens must equal the lens stage's (results/r6/lens) to rtol: z*
    norm-wise (‖Δz*‖ ≤ rtol·‖z*‖; elementwise, a near-zero component of a vector solved
    from an ill-conditioned A fails on rounding alone), ‖c⊥‖ and κ as scalars, the
    principal widths elementwise. The observed relative differences are returned."""
    p = os.path.join(ROOT, cfg["paths"]["lens_out"], f"{name_of(task, seed)}.npz")
    ref = np.load(p)
    diff = dict(z_star=float(np.linalg.norm(L.z_star - ref["z_star"]) / np.linalg.norm(ref["z_star"])),
                norm_c_perp=float(abs(L.norm_c_perp - ref["norm_c_perp"]) / abs(ref["norm_c_perp"])),
                kappa=float(abs(L.kappa - ref["kappa"]) / abs(ref["kappa"])),
                principal_widths=float(np.max(np.abs(L.principal_widths - ref["principal_widths"])
                                              / np.abs(ref["principal_widths"]))))
    bad = [k for k, v in diff.items() if not v <= rtol]
    if bad:
        die(f"{task} seed {seed}: lens differs from {p} in {bad}: relative differences {diff}.")
    return dict(file=os.path.relpath(p, ROOT).replace(os.sep, "/"), sha256=pv.sha256(p),
                rel_diff=diff, rtol=rtol)


def tdmpc2_init_record(cfg):
    """D7 (4): "The meta also records tdmpc2's initialisation code
    (`tdmpc2/common/init.py` at tdmpc2 commit e9f59321): its path, commit and SHA-256."
    Read from the commit itself (git show), not the working tree."""
    src = os.path.join(ROOT, cfg["paths"]["tdmpc2_src"])
    commit, path = cfg["source"]["repo_commit"], cfg["criterion"]["tdmpc2_init"]
    if not os.path.isdir(os.path.join(src, ".git")):
        die(f"tdmpc2 source not found at {src}. Clone it first:\n"
            f"  git clone {cfg['source']['repo']} {cfg['paths']['tdmpc2_src']}\n"
            f"  git -C {cfg['paths']['tdmpc2_src']} checkout {commit}")
    r = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, cwd=src)
    if r.returncode != 0:
        die(f"git show {commit}:{path} in {src} failed: {r.stderr.decode('utf-8', 'replace').strip()}")
    import hashlib
    return dict(repo=cfg["source"]["repo"], commit=commit, path=path,
                sha256=hashlib.sha256(r.stdout).hexdigest(), text=r.stdout.decode("utf-8"))


# --------------------------------------------------------------------------- #
# outputs                                                                       #
# --------------------------------------------------------------------------- #


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


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def write_outputs(out_dir, res):
    """Every result file of run_stages; after a stop, only the Corollary 1 rows."""
    write_csv(os.path.join(out_dir, "corollary1.csv"), res["cor1_rows"])
    if res["status"] == STATUS_STOPPED:
        return
    s2 = res["s2"]
    write_csv(os.path.join(out_dir, "consistency_planner.csv"), s2["cons_rows"])
    if s2["d4_rows"]:
        write_csv(os.path.join(out_dir, "consistency_d4.csv"), s2["d4_rows"])
    write_json(os.path.join(out_dir, "identification.json"), s2["ident"])
    write_csv(os.path.join(out_dir, "criterion.csv"), res["crit_rows"])
    write_csv(os.path.join(out_dir, "rho_fractions.csv"), res["rho_rows"])
    write_json(os.path.join(out_dir, "g1.json"), dict(status=res["status"], **res["g1"]))
    write_csv(os.path.join(out_dir, "g1.csv"), res["g1"]["rows"])
    write_csv(os.path.join(out_dir, "susceptibility.csv"), res["sus_rows"])
    write_json(os.path.join(out_dir, "borderline.json"),
               dict(status=res["status"], band=res["band"], items=res["borderline"]))
    for name, arrays in res["lines"].items():
        os.makedirs(os.path.join(out_dir, "lines"), exist_ok=True)
        np.savez(os.path.join(out_dir, "lines", f"{name}.npz"), **arrays)
    for ((task, seed), reading), (idx, nn) in res["pop"].items():
        os.makedirs(os.path.join(out_dir, "populated"), exist_ok=True)
        np.savez(os.path.join(out_dir, "populated", f"{name_of(task, seed)}-{reading}.npz"),
                 subsample_index=idx, nearest_other=nn)


def to_drive(local, drive):
    """Copy one result file to Drive under the same relative path, and verify it."""
    dst = drive_path(drive, os.path.relpath(local, ROOT).replace(os.sep, "/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(local, dst + ".part")
    os.replace(dst + ".part", dst)
    if pv.sha256(dst) != pv.sha256(local):
        die(f"the Drive copy of {local} does not verify.")


def banner(lines):
    w = max(len(s) for s in lines) + 4
    print("\n" + "#" * w)
    for s in lines:
        print(f"# {s.ljust(w - 4)} #")
    print("#" * w)


def summary(res):
    if res["status"] == STATUS_STOPPED:
        bad = [r for r in res["cor1_rows"] if r.get("status") == "FAIL"]
        banner([f"STOPPED: {STATUS_STOPPED}", f"{len(bad)} line(s) above 1e-10; no criterion result was",
                "computed. Consult the author (D7 (4)). See corollary1.csv."])
        return
    s2, g1 = res["s2"], res["g1"]
    print("\n================ results ================")
    for r in res["crit_rows"]:
        print(f"  {r['task']} seed {r['seed']} [{r['reading']}{'' if r['primary'] else ', reported'}]: "
              f"Inside {r['inside']} (w2 {r['inside_w2']:.4g} vs {r['inside_chi2_q']:.4g}), "
              f"Populated {r['populated']} ({r['populated_dist']:.4g} vs {r['populated_threshold']:.4g}), "
              f"Sharp {r['sharp']} ({r['sharp_ratio']:.4g}) -> {r['criterion']}"
              f"{'  FLAG < 0.5' if r['flag_below_half'] else ''}")
    g = s2["gate"]
    print(f"\ncartpole-swingup seed 1: condition (i) {g['condition_i']} (fraction {g['fraction']:.3f}), "
          f"condition (ii) {g['condition_ii']} (e/e0 {s2['cii_gate']['e_over_e0']:.4g}) -> identified "
          f"{g['identified']}; tie ratios {s2['tie']['ratios']}, within 10%: {s2['tie']['within']}")
    print(f"humanoid-run seed 3: {s2['rep']['label']}")
    print("\nG1:")
    for r in g1["rows"]:
        print(f"  {r['task']:18s} seed {r['counting_seed']} ({r['why']}): criterion {r['criterion']}"
              f"{'' if r['counts'] else '  (flagged: leaves G1)'}")
    print(f"  {g1['n_pass']} of {g1['n_counting']} pass; G1 needs {g1['min_pass']}: "
          f"{'PASSES' if g1['g1'] else 'does not pass'}")
    if res["status"] == STATUS_PENDING:
        hits = [b for b in res["borderline"] if b["borderline"]]
        banner(["PENDING COLAB RERUN (D7 (8)): these results are not final.",
                f"{len(hits)} statistic(s) within 1% of the threshold:"]
               + [f"  {b['where']}: {b['statistic']}: {b['value']:.6g} vs {b['threshold']:.6g}" for b in hits]
               + ["Rerun the stage in the Colab CPU environment of meta_d4_regeneration.json;",
                  "where the two disagree on pass or fail, D7 (8)'s rules decide."])


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #


def versions():
    out = {"python": platform.python_version()}
    for p in ("jax", "jaxlib", "numpy", "scipy", "torch", "PyYAML"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--drive", required=True, help="the Drive folder layernorm-lens-r6")
    ap.add_argument("--out", default=None, help="output directory (default: the config's criterion.results)")
    args = ap.parse_args()
    try:
        pv.require_prereg()
        origin_tags = pv.require_tags_on_origin()
        table = pv.d6_sha_table()
    except pv.ProvenanceError as e:
        die(str(e))
    state = pv.git_state()
    if state["git_dirty"]:
        die("refusing to run: the working tree has uncommitted changes to tracked files.")
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cc = cfg["criterion"]
    drive = os.path.abspath(args.drive)
    if not os.path.isdir(drive):
        die(f"--drive {drive!r} is not a directory.")
    out_dir = os.path.join(ROOT, *(args.out or cc["results"]).split("/"))
    stamp = datetime.datetime.now(datetime.timezone.utc)
    init = tdmpc2_init_record(cfg)

    checkpoints = [(t, s, cfg["survey"][t][s]) for t in cfg["tasks"] for s in cfg["seeds"]]
    planner, planner_rec = planner_inputs(cfg, drive, checkpoints)
    idc = cc["identification"]
    pre = [(idc["gate"]["task"], int(idc["gate"]["seed"])), (idc["reported"]["task"], int(idc["reported"]["seed"]))]
    d4, d4_rec = d4_inputs(cfg, drive, pre)
    cks, ck_rec, lens_rec = {}, {}, {}
    for task, seed, layout in checkpoints:
        n = name_of(task, seed)
        print(f"loading {n} [{layout}]", flush=True)
        sd, mods, sha = load_checkpoint(cfg, table, task, seed)
        if layouts.detect_layout(sd, n) != layout:
            die(f"{n}: detected layout differs from the survey (D5).")
        O, A, frac, flag = planner[(task, seed)]
        cks[(task, seed)] = Checkpoint(task, seed, layout, sd, mods, O, A, frac, flag)
        ck_rec[pv.checkpoint_name(task, seed)] = sha
        lin, _ = layouts.check_first_layer(sd, layout, cfg["layer"]["layouts"], n)
        lens_rec[n] = check_lens(cfg, geo.lens(sd[f"{lin}.weight"], sd[f"{lin}.bias"],
                                               float(cfg["layer"]["layernorm_eps"])), task, seed)

    res = run_stages(cfg, cks, d4)
    os.makedirs(out_dir, exist_ok=True)
    write_outputs(out_dir, res)
    meta = dict(stage="criterion", status=res["status"], **state, prereg_tags=pv.tag_objects(),
                prereg_tags_on_origin=origin_tags, versions=versions(), platform=platform.platform(),
                time_start=stamp.isoformat(), time_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                d6_table_source=f"git show {pv.D6_TAG}:{pv.D6_PATH}", tdmpc2_init=init,
                inputs=dict(checkpoints=ck_rec, planner=planner_rec, d4=d4_rec, lens=lens_rec),
                outputs=OUTPUTS, config=cfg)
    write_json(os.path.join(out_dir, "meta_criterion.json"), meta)

    files = sorted(p for p in glob.glob(os.path.join(out_dir, "**", "*"), recursive=True) if os.path.isfile(p))
    big = [p for p in files if os.path.getsize(p) >= 1 << 20]
    commit = [p for p in files if p not in big]
    for p in files:
        to_drive(p, drive)
    missing = [p for p in commit if pv.sha256(drive_path(drive, os.path.relpath(p, ROOT).replace(os.sep, "/")))
               != pv.sha256(p)]
    if missing:
        die(f"Drive copies do not verify: {missing}")
    summary(res)
    if big:
        print(f"WARNING: over 1 MB, not listed for commit: {[os.path.relpath(p, ROOT) for p in big]}")
    print(f"\nall {len(files)} output files are on Drive with matching SHA-256.")
    print("files to commit:")
    print(pv.git_add_command(commit))
    if res["status"] == STATUS_STOPPED:
        sys.exit(2)


if __name__ == "__main__":
    main()
