"""R6 criterion stage: r6.md's criterion and G1, D6 (b) identification, D6 (c)
calibration and D6 (g) susceptibility, on the D6 planner data.

    python experiments/r6_tdmpc2/criterion.py --config experiments/r6_tdmpc2/config.yaml --drive DIR

Runs on a Colab CPU runtime in the project's environment (JAX float64; torch only to
read the .pt checkpoints). Refuses to run unless the annotated tags prereg-r6, -d1,
-d2 and -d3 exist and the four pre-registration files match them.

Inputs, each checked before use:
  - checkpoints: downloaded from the pinned Hugging Face revision, refused unless the
    SHA-256 matches D6's table, read from the tag prereg-r6-d3 (provenance.py);
  - planner observations (MyDrive/layernorm-lens-r6/data/r6/planner/): refused unless
    each file's SHA-256 matches its row of results/r6/planner/planner_obs_manifest.csv
    (kind "data"; smoke files are never read) and the checkpoint's meta in
    results/r6/planner/ (complete, 50 episodes, the same SHA-256). For the two
    pre-release checkpoints the meta must record that the controls behaved as
    stated and the encoder agreement gate passed;
  - D4 observations (record-only D3 values, D6 (b)): as planner_collect.verify_d4,
    match: true in meta_d4_regeneration.json and each file's SHA-256 as in
    d4_obs_manifest.csv;
  - each checkpoint's lens is recomputed from its weights and must equal the lens
    stage's results/r6/lens/<task>-seed<seed>.npz (z*, ‖c⊥‖, κ, principal widths).

Coordinates (D6 (b)): what the layer receives. Identity for the 13 public-layout
checkpoints; symlog(o) = sign(o)·log(1 + |o|) for cartpole-swingup seed 1 and
humanoid-run seed 3, computed in float64 by layouts.input_candidates(...)["symlog"]
from the stored raw observations, whatever label (b) gives them.

Per checkpoint (all 15): r6.md's Inside, Populated and Sharp and their conjunction
(criterion_lib.py quotes each definition), the quantities r6.md reports regardless
of outcome that need data (ρ and ρ_eff fractions), and D6 (g) along d₁, d₂ and
u_min for the encoder output (after SimNorm) and the dynamics prediction (a = 0).
Then: D3's computation on the planner data for the pre-release checkpoints and the
calibration checkpoints (D6 (c)), condition (ii) on the D4 data (record only), the
D6 (b) identification and label, and G1 from the counting checkpoints.

Outputs, in results/r6/criterion/ (copied to Drive under the same path): see
OUTPUTS below and meta_criterion.json. The run ends by printing the exact
`git add -f` command for them.
"""

import argparse
import csv
import datetime
import glob
import importlib.metadata as md
import json
import os
import platform
import shutil
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
OUTPUTS = """
criterion.csv              one row per checkpoint: coordinates, n, k, m, Inside, Populated,
                           Sharp, criterion, return fraction and the < 0.5 flag
rho_fractions.csv          r6.md, reported regardless: ρ_eff and ρ fractions
consistency_planner.csv    D3 on the planner data (pre-release and D6 (c) calibration)
consistency_d4.csv         D3 on the D4 data for the pre-release checkpoints (record only)
identification.json        D6 (b): cartpole-swingup seed 1 conditions (i), (ii); humanoid-run
                           seed 3 label, next to the humanoid-run seed 1 calibration value
g1.json, g1.csv            G1: counting checkpoint per task, its result, the vote
susceptibility.csv         D6 (g): per checkpoint, direction and output
lines/<name>.npz           D6 (g): t grid, ‖J(t)‖ and ‖g′‖ along each line (float64)
populated/<name>.npz       Populated: subsample indices and nearest-other distances
meta_criterion.json        config, commit, tags, versions, every input file's SHA-256
"""


def die(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def name_of(task, seed):
    return f"{task}-seed{seed}"


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def to_drive(local, drive):
    """Copy one result file to Drive under the same relative path, and verify it."""
    dst = os.path.join(drive, os.path.relpath(local, ROOT))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(local, dst + ".part")
    if pv.sha256(dst + ".part") != pv.sha256(local):
        die(f"copy of {local} to Drive does not verify.")
    os.replace(dst + ".part", dst)


# --------------------------------------------------------------------------- #
# inputs                                                                        #
# --------------------------------------------------------------------------- #


def planner_record(cfg, checkpoints):
    """The planner run's manifest and metas: refuse anything incomplete or inconsistent."""
    cc = cfg["criterion"]
    pdir = os.path.join(ROOT, cc["planner_results"])
    man_path = os.path.join(ROOT, cc["planner_manifest"])
    rows = [r for r in pv.read_manifest(man_path) if r["kind"] == "data"]
    by_file = {os.path.basename(r["file"]): r for r in rows}
    want = len(cfg["planner_collect"]["env_seeds"]) * int(cfg["planner_collect"]["episodes_per_env_seed"])
    rec = {}
    for task, seed, layout in checkpoints:
        n = name_of(task, seed)
        mp = os.path.join(pdir, f"meta_{n}.json")
        if not os.path.exists(mp):
            die(f"{mp} not found: no planner data for {n}.")
        m = json.load(open(mp))
        if not (m.get("complete") is True and not m.get("smoke") and m["kind"] == "data"
                and m["returns"]["episodes"] == want and m["eval_mode"] is True):
            die(f"{n}: planner meta is not a complete 50-episode eval_mode=True data run.")
        if m["layout"] != layout or m["task"] != task or m["seed"] != seed:
            die(f"{n}: planner meta records {m['task']} seed {m['seed']} [{m['layout']}].")
        row = by_file.get(f"{n}.npz")
        if row is None or row["sha256"] != m["data"]["sha256"]:
            die(f"{n}: manifest row {row} does not match the meta's SHA-256 {m['data']['sha256']}.")
        want_input = "symlog" if layout == layouts.PRERELEASE else "identity"
        if m["planner_input"] != want_input:
            die(f"{n}: planner acted on {m['planner_input']}, D6 (b) requires {want_input}.")
        if layout == layouts.PRERELEASE:
            if not (m.get("controls") or {}).get("as_stated") or not (m.get("gate_result") or {}).get("pass"):
                die(f"{n}: its meta does not record controls as stated and a passed gate.")
        rec[(task, seed)] = dict(meta=m, meta_file=os.path.relpath(mp, ROOT), manifest_row=row)
    return rec, dict(manifest=os.path.relpath(man_path, ROOT), manifest_sha256=pv.sha256(man_path))


def load_planner_obs(drive, rec):
    path = os.path.join(drive, rec["manifest_row"]["file"])
    if not os.path.exists(path):
        die(f"planner observations {path} not found on Drive.")
    try:
        pv.verify_sha256(path, rec["manifest_row"]["sha256"], rec["manifest_row"]["file"])
    except pv.ProvenanceError as e:
        die(str(e))
    d = np.load(path)
    return d["obs"], d["actions"], d["rewards"]


def load_checkpoint(cfg, table, task, seed):
    """Download and verify (D6 table, from the tag), then a float64 numpy state_dict."""
    import torch
    name = pv.checkpoint_name(task, seed)
    repo = cfg["source"]["checkpoints"].rstrip("/").split("huggingface.co/")[1]
    url = pv.HF_FILE.format(repo=repo, rev=cfg["source"]["hf_revision"], path=name)
    local = os.path.join(ROOT, cfg["paths"]["checkpoints"], name)
    try:
        sha = pv.download_verified(url, local, table[name], name)
    except pv.ProvenanceError as e:
        die(str(e))
    ck = torch.load(local, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    mods = getattr(sd, "_metadata", None)
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in sd.items() if hasattr(v, "detach")}
    return sd, (None if mods is None else sorted(mods.keys())), sha


def networks(cfg, sd, mods, layout, cand):
    """numpy networks as D3 builds them (collect.agent), with D3 candidate `cand` at
    the encoder's input."""
    c = cfg["consistency"]
    eps, sdim = float(cfg["layer"]["layernorm_eps"]), int(c["simnorm_dim"])
    inp = layouts.input_candidates(float(c["pos0_layernorm_eps"]))[cand]
    enc, dyn, pi = layouts.build_networks(sd, layout, eps, sdim, mods, input_fn=inp)
    return dict(enc=enc, dyn=dyn, pi=pi)


def check_lens(cfg, L, task, seed):
    """The recomputed lens must equal the lens stage's (results/r6/lens)."""
    p = os.path.join(ROOT, cfg["paths"]["lens_out"], f"{name_of(task, seed)}.npz")
    ref = np.load(p)
    pairs = dict(z_star=L.z_star, norm_c_perp=L.norm_c_perp, kappa=L.kappa,
                 principal_widths=L.principal_widths)
    bad = [k for k, v in pairs.items() if not np.allclose(v, ref[k], rtol=1e-12, atol=0)]
    if bad:
        die(f"{task} seed {seed}: lens differs from {p} in {bad}.")
    return dict(file=os.path.relpath(p, ROOT), sha256=pv.sha256(p))


# --------------------------------------------------------------------------- #
# one checkpoint                                                                #
# --------------------------------------------------------------------------- #


def run_checkpoint(cfg, task, seed, layout, sd, mods, O, A, out_dir):
    cc = cfg["criterion"]
    eps, sdim = float(cfg["layer"]["layernorm_eps"]), int(cfg["consistency"]["simnorm_dim"])
    coords = "symlog" if layout == layouts.PRERELEASE else "identity"
    to_layer = layouts.input_candidates(float(cfg["consistency"]["pos0_layernorm_eps"]))[coords]
    X = to_layer(np.asarray(O, np.float64).reshape(-1, O.shape[-1]))
    lin, _ = layouts.check_first_layer(sd, layout, cfg["layer"]["layouts"], f"{task} seed {seed}")
    E, b = sd[f"{lin}.weight"], sd[f"{lin}.bias"]
    L = geo.lens(E, b, eps)

    S = cl.data_stats(X, float(cc["kept_rtol"]))
    crit, idx = cl.criterion(S, X, L, cc)
    row = dict(task=task, seed=seed, layout=layout, coordinates=coords, n=S.n, k=S.k,
               lambda1=float(S.eigvals[0]), lambda2=float(S.eigvals[1]), d2_kept=S.d2_kept,
               norm_z_star=float(np.linalg.norm(L.z_star)), kappa=float(L.kappa),
               degenerate=bool(L.degenerate), **crit)
    rho = dict(task=task, seed=seed, **cl.rho_fractions(L, X, cc["rho_levels"]))

    # D6 (g): networks from layer-input coordinates (identity at the encoder input)
    num = networks(cfg, sd, mods, layout, "identity")
    enc_j = cl.jax_mlp(cl.mlp_spec(sd, layout, "_encoder.state", "simnorm"), eps, sdim)
    dyn_j = cl.jax_mlp(cl.mlp_spec(sd, layout, "_dynamics", "simnorm"), eps, sdim)
    a_dim = A.shape[-1]
    probe = X[np.random.default_rng(0).choice(len(X), 256, replace=False)]
    zp = np.concatenate([num["enc"](probe), np.zeros((len(probe), a_dim))], -1)
    for what, f_np, f_j, x in (("encoder", num["enc"], enc_j, probe), ("dynamics", num["dyn"], dyn_j, zp)):
        dev = np.max(np.abs(f_np(x) - np.asarray(f_j(jnp.asarray(x)))))
        if dev > 1e-12:
            die(f"{task} seed {seed}: JAX {what} differs from layouts.py by {dev:.3g}.")

    def enc_f(x):
        return enc_j(x)

    def dyn_a0(x):
        return dyn_j(jnp.concatenate([enc_j(x), jnp.zeros(a_dim)]))

    def dyn_rec(xa):
        return dyn_j(jnp.concatenate([enc_j(xa[:S.k]), xa[S.k:]]))

    Xa = np.concatenate([X.reshape(O.shape[0], O.shape[1], -1)[:, :-1].reshape(-1, S.k),
                         np.asarray(A, np.float64).reshape(-1, a_dim)], -1)
    sus_rows, lines = [], {}
    dirs = dict(d1=S.d1, d2=S.d2 if S.d2_kept else None,
                u_min=cl.canonical_sign(L.principal_dirs[:, int(np.argmin(L.principal_widths))]))
    for dname, d in dirs.items():
        if d is None:
            sus_rows.append(dict(task=task, seed=seed, direction=dname, status="degenerate (D6 (g))"))
            continue
        if L.degenerate:
            sus_rows.append(dict(task=task, seed=seed, direction=dname, status="degenerate lens"))
            continue
        dpad = np.concatenate([d, np.zeros(a_dim)])
        outs = {"encoder": (enc_f, {"a0": (enc_f, X, d)}),
                "dynamics": (dyn_a0, {"a0": (dyn_a0, X, d), "recorded": (dyn_rec, Xa, dpad)})}
        r = cl.susceptibility(L, E, b, d, X, outs, int(cc["susceptibility"]["n_grid"]),
                              float(cc["susceptibility"]["t_range_r_eff"]),
                              float(cc["susceptibility"]["cor1_rtol"]))
        if not r["cor1_ok"]:
            die(f"{task} seed {seed} {dname}: Corollary 1 check failed, max relative deviation "
                f"{r['cor1_max_rel_dev']:.3g} > {cc['susceptibility']['cor1_rtol']}.")
        lines[f"{dname}_t"] = r["t"]
        for oname, o in r["out"].items():
            lines[f"{dname}_{oname}_J"] = o.pop("J_line")
            lines[f"{dname}_{oname}_g_prime"] = o.pop("g_prime_line")
            sus_rows.append(dict(task=task, seed=seed, direction=dname, output=oname, status="ok",
                                 r_star=r["r_star"], r_eff=r["r_eff"], T=r["T"],
                                 cor1_max_rel_dev=r["cor1_max_rel_dev"], **o))
    os.makedirs(os.path.join(out_dir, "lines"), exist_ok=True)
    np.savez(os.path.join(out_dir, "lines", f"{name_of(task, seed)}.npz"), **lines,
             d1=S.d1, d2=S.d2 if S.d2 is not None else np.full(S.k, np.nan),
             u_min=dirs["u_min"], z_star=L.z_star)
    os.makedirs(os.path.join(out_dir, "populated"), exist_ok=True)
    np.savez(os.path.join(out_dir, "populated", f"{name_of(task, seed)}.npz"), subsample_index=idx)
    return row, rho, sus_rows, L


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--drive", required=True, help="the Drive folder MyDrive/layernorm-lens-r6")
    args = ap.parse_args()
    try:
        pv.require_prereg()
        table = pv.d6_sha_table()
    except pv.ProvenanceError as e:
        die(str(e))
    cfg = yaml.safe_load(open(args.config))
    cc = cfg["criterion"]
    drive = os.path.abspath(args.drive)
    if not os.path.isdir(drive):
        die(f"--drive {drive!r} is not a directory (mount Google Drive first).")
    out_dir = os.path.join(ROOT, cc["results"])
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc)

    checkpoints = [(t, s, cfg["survey"][t][s]) for t in cfg["tasks"] for s in cfg["seeds"]]
    record, manifest_info = planner_record(cfg, checkpoints)
    inputs = dict(planner=manifest_info, checkpoints={}, planner_obs={}, lens={}, d4=None)

    crit_rows, rho_rows, sus_rows, nets, data = [], [], [], {}, {}
    for task, seed, layout in checkpoints:
        n = name_of(task, seed)
        print(f"\n=== {n} [{layout}] ===", flush=True)
        sd, mods, sha = load_checkpoint(cfg, table, task, seed)
        if layouts.detect_layout(sd, n) != layout:
            die(f"{n}: detected layout differs from the survey (D5).")
        O, A, R = load_planner_obs(drive, record[(task, seed)])
        row, rho, sus, L = run_checkpoint(cfg, task, seed, layout, sd, mods, O, A, out_dir)
        inputs["lens"][n] = check_lens(cfg, L, task, seed)
        m = record[(task, seed)]["meta"]
        row.update(fraction=m["returns"]["fraction"], flag_below_half=m["returns"]["flag_below_half"])
        crit_rows.append(row), rho_rows.append(rho), sus_rows.extend(sus)
        inputs["checkpoints"][n] = sha
        inputs["planner_obs"][n] = record[(task, seed)]["manifest_row"]["sha256"]
        nets[(task, seed)] = (sd, mods, layout)
        data[(task, seed)] = (O, A)
        print(f"  Inside {row['inside']} (w² {row['inside_w2']:.4g} vs {row['inside_chi2_q']:.4g}, m {row['m']}); "
              f"Populated {row['populated']} ({row['populated_dist']:.4g} vs {row['populated_threshold']:.4g}); "
              f"Sharp {row['sharp']} (D/r_eff {row['sharp_ratio']:.4g}) -> criterion {row['criterion']}")
        write_csv(os.path.join(out_dir, "criterion.csv"), crit_rows)
        write_csv(os.path.join(out_dir, "rho_fractions.csv"), rho_rows)
        write_csv(os.path.join(out_dir, "susceptibility.csv"), sus_rows)
        for p in glob.glob(os.path.join(out_dir, "**", "*"), recursive=True):
            if os.path.isfile(p):
                to_drive(p, drive)

    # D3's computation on the planner data (D6 (b), (c)) and on the D4 data (record only)
    c = cfg["consistency"]
    idc = cc["identification"]
    jobs = ([(t["task"], int(t["seed"]), "prerelease") for t in (idc["gate"], idc["reported"])]
            + [(t["task"], int(t["seed"]), "calibration") for t in cc["calibration"]])
    cons_rows, err_planner = [], {}
    for task, seed, role in jobs:
        sd, mods, layout = nets[(task, seed)]
        ags = {cand: networks(cfg, sd, mods, layout, cand) for cand in c["candidates"]}
        O, A = data[(task, seed)]
        err = collect.consistency_errors(ags, O, A, int(c["rng_seed"]))
        err_planner[(task, seed)] = err
        for cand, x in err.items():
            cons_rows.append(dict(task=task, seed=seed, role=role, layout=layout, data="planner",
                                  candidate=cand, e=x["e"], e0=x["e0"], e_over_e0=x["e"] / x["e0"], n=x["n"]))
    write_csv(os.path.join(out_dir, "consistency_planner.csv"), cons_rows)

    import planner_collect  # verify_d4: match: true and each file as in d4_obs_manifest.csv
    pre = [(idc["gate"]["task"], int(idc["gate"]["seed"])), (idc["reported"]["task"], int(idc["reported"]["seed"]))]
    d4_paths, inputs["d4"] = planner_collect.verify_d4(cfg, drive, pre)
    d4_rows, err_d4 = [], {}
    for task, seed in pre:
        sd, mods, layout = nets[(task, seed)]
        ags = {cand: networks(cfg, sd, mods, layout, cand) for cand in c["candidates"]}
        d = np.load(d4_paths[(task, seed)])
        err = collect.consistency_errors(ags, d["obs"], d["actions"], int(c["rng_seed"]))
        err_d4[(task, seed)] = err
        for cand, x in err.items():
            d4_rows.append(dict(task=task, seed=seed, data="D4 policy prior (record only)", candidate=cand,
                                e=x["e"], e0=x["e0"], e_over_e0=x["e"] / x["e0"], n=x["n"]))
    write_csv(os.path.join(out_dir, "consistency_d4.csv"), d4_rows)

    # D6 (b)
    frac = {k: v["meta"]["returns"]["fraction"] for k, v in record.items()}
    cii = {k: cl.condition_ii(err_planner[k], idc["candidate"], float(idc["e_over_e0_max"])) for k in pre}
    cii_d4 = {k: cl.condition_ii(err_d4[k], idc["candidate"], float(idc["e_over_e0_max"])) for k in pre}
    gate, rep = cl.identification(frac[pre[0]], cii[pre[0]], frac[pre[1]], cii[pre[1]],
                                  float(idc["return_ratio"]))
    h1 = err_planner[("humanoid-run", 1)]
    ident = {name_of(*pre[0]): dict(**gate, condition_ii_detail=cii[pre[0]],
                                    condition_ii_on_d4_record_only=cii_d4[pre[0]]),
             name_of(*pre[1]): dict(**rep, condition_ii_detail=cii[pre[1]],
                                    condition_ii_on_d4_record_only=cii_d4[pre[1]],
                                    humanoid_run_seed1_calibration_e_over_e0={
                                        k: v["e"] / v["e0"] for k, v in h1.items()})}
    write_json(os.path.join(out_dir, "identification.json"), ident)
    for k in pre:
        if cii[k]["within_1p1"]:
            print(f"NOTE {name_of(*k)}: {cii[k]['within_1p1']} within 10% of symlog's e on the planner data.")

    # G1
    res = {(r["task"], r["seed"]): r["criterion"] for r in crit_rows}
    flags = {(r["task"], r["seed"]): r["flag_below_half"] for r in crit_rows}
    g1 = cl.g1_vote(res, gate["identified"], flags, cc["g1"]["tasks"], int(cc["g1"]["min_pass"]))
    write_json(os.path.join(out_dir, "g1.json"), g1)
    write_csv(os.path.join(out_dir, "g1.csv"), g1["rows"])

    # meta, Drive, summary
    vers = {}
    for p in ("jax", "jaxlib", "numpy", "scipy", "torch", "PyYAML"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    meta = dict(stage="criterion", **pv.git_state(), prereg_tags=pv.tag_objects(),
                python=platform.python_version(), versions=vers, jax=jax.__version__,
                platform=platform.platform(), time_start=stamp.isoformat(),
                time_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                inputs=inputs, d6_table_source=f"git show {pv.D6_TAG}:{pv.D6_PATH}",
                outputs=OUTPUTS, config=cfg)
    write_json(os.path.join(out_dir, "meta_criterion.json"), meta)
    files = [p for p in glob.glob(os.path.join(out_dir, "**", "*"), recursive=True) if os.path.isfile(p)]
    for p in files:
        to_drive(p, drive)

    print("\n================ results ================")
    for t in cfg["tasks"]:
        print(f"{t}:")
        for r in (r for r in crit_rows if r["task"] == t):
            print(f"  seed {r['seed']} [{r['coordinates']}]: Inside {r['inside']}, Populated "
                  f"{r['populated']}, Sharp {r['sharp']} -> {r['criterion']}"
                  f"{'  FLAG < 0.5' if r['flag_below_half'] else ''}")
    print(f"\ncartpole-swingup seed 1: condition (i) {gate['condition_i']} (fraction {gate['fraction']:.3f}), "
          f"condition (ii) {gate['condition_ii']} -> identified {gate['identified']}")
    print(f"humanoid-run seed 3: {rep['label']}")
    print("\nG1:")
    for r in g1["rows"]:
        print(f"  {r['task']:18s} seed {r['counting_seed']} ({r['why']}): criterion {r['criterion']}"
              f"{'' if r['counts'] else '  (flagged: leaves G1)'}")
    print(f"  {g1['n_pass']} of {g1['n_counting']} pass; G1 needs {g1['min_pass']}: "
          f"{'PASSES' if g1['g1'] else 'does not pass'}")
    big = [p for p in files if os.path.getsize(p) >= 1 << 20]
    if big:
        print(f"WARNING: over 1 MB, not listed for commit: {[os.path.relpath(p, ROOT) for p in big]}")
    print("\nfiles to commit:")
    print(pv.git_add_command([p for p in files if p not in big]))


if __name__ == "__main__":
    main()
