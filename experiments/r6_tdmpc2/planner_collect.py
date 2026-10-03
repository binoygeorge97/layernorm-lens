"""R6 deviation D6 (a), (b), (e): planner data for all 15 checkpoints, with the
encoder agreement gate and its controls, through the public tdmpc2 code (e9f59321).

    python experiments/r6_tdmpc2/planner_collect.py --config experiments/r6_tdmpc2/config.yaml --drive DIR [--smoke]

Runs in tdmpc2's pinned Python 3.11 environment on a CUDA GPU (a Colab T4), like
planner_check.py. The planner runs in float32 PyTorch for acting only. This entry
point imports no JAX (tdmpc2's environment has none), so CLAUDE.md's x64 rule is met
by doing every computation of ours (lens-side encoder, relative errors, d1, z*,
r_eff) in float64 numpy.

Refuses to run unless the annotated tags prereg-r6, -d1, -d2 and -d3 exist and the
four pre-registration files match them. Computes no criterion quantity (Inside,
Populated, Sharp), no G1, no consistency (c) and no susceptibility (g).

Order (D6 (b), (e); the author's halting rule of 25 Sep 2026):
 1. Checkpoints: each downloaded from the pinned Hugging Face revision and refused
    unless its SHA-256 matches D6's table, read from the tag prereg-r6-d3.
 2. D4 observations (MyDrive/layernorm-lens-r6/data/r6/): used only if
    results/r6/meta_d4_regeneration.json says match: true and each file's SHA-256
    matches results/r6/d4_obs_manifest.csv.
 3. Controls (positive: cartpole-swingup seed 2, public, identity on both sides;
    negative: cartpole-swingup seed 1 remapped, identity on the planner side, symlog
    on the lens side), then the encoder agreement gate (both pre-release checkpoints,
    remap + symlog), on the loaded TDMPC2 objects that later act. A control that
    does not behave as stated halts everything, public collection included (exit
    4). A failed gate halts only the pre-release runs (exit 5 at the end).
 4. The 13 public-layout checkpoints, then the 2 pre-release ones: 50 episodes each
    (environment seeds 0-4; make_env with seed = s and set_seed(s) once before its
    10 episodes), eval_mode=True, 500 agent steps (action repeat 2), all 501 raw
    observations, 500 executed actions and 500 rewards per episode. The pre-release
    ones act on symlog(o) (layouts.input_candidates, float64, cast to float32).

Resumable: a checkpoint whose data file and meta are on Drive, with the data file's
SHA-256 equal to the one in its meta, is skipped. Controls run in every session that
has work left; the gate whenever a pre-release run is left. After each checkpoint,
its outputs are copied to Drive and the summaries (planner_returns.csv,
planner_obs_manifest.csv, encoder_gate.csv) are rewritten. Every session ends with
the exact `git add -f` command for the result files. Observation data is never
committed.

--smoke: controls and gate, then 1 episode (environment seed 0) of cartpole-swingup
seed 1 if the gate passed. Its outputs are outcome data: kept on Drive under
planner_smoke/, listed in the manifest (kind "smoke") and the summary, never used for
any rule and never counted toward the 50 episodes.
"""

import argparse
import dataclasses
import datetime
import glob
import importlib.metadata as md
import json
import os
import platform
import shutil
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import layouts  # noqa: E402
import planner_lib as pl  # noqa: E402
import provenance as pv  # noqa: E402
from planner_check import published_return, tdmpc2_cfg  # noqa: E402

PACKAGES = ["torch", "tensordict", "torchrl", "mujoco", "dm_control", "numpy", "gymnasium",
            "hydra-core", "omegaconf", "PyYAML"]
EXIT_CONTROL, EXIT_GATE = 4, 5


def die(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def utc():
    return datetime.datetime.now(datetime.timezone.utc)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_csv(path, rows, fields=None):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def name_of(task, seed):
    return f"{task}-seed{seed}"


# --------------------------------------------------------------------------- #
# locations                                                                     #
# --------------------------------------------------------------------------- #


class Paths:
    """Local and Drive locations of data and results (normal run or smoke)."""

    def __init__(self, cfg, drive, smoke):
        d = cfg["planner_collect"]["dirs"]
        self.dirs = d
        self.drive = drive
        self.kind = "smoke" if smoke else "data"
        self.data_rel = d["smoke_data"] if smoke else d["data"]
        self.res_rel = d["smoke_results"] if smoke else d["results"]
        self.data = os.path.join(ROOT, self.data_rel)
        self.res = os.path.join(ROOT, self.res_rel)
        self.drive_data = os.path.join(drive, self.data_rel)
        self.drive_res = os.path.join(drive, self.res_rel)
        # summaries always live with the normal run's results
        self.summary = os.path.join(ROOT, d["results"])
        self.drive_summary = os.path.join(drive, d["results"])
        for p in (self.data, self.res, self.summary):
            os.makedirs(p, exist_ok=True)


def copy_verified(src, dst, overwrite=False):
    """Copy and verify; never replace a different file unless overwrite (summaries)."""
    h = pv.sha256(src)
    if os.path.exists(dst) and pv.sha256(dst) == h:
        return h
    if os.path.exists(dst) and not overwrite:
        die(f"{dst} exists with a different SHA-256; not overwriting.")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".part")
    if pv.sha256(dst + ".part") != h:
        die(f"copy of {src} to {dst} does not verify.")
    os.replace(dst + ".part", dst)
    return h


def pull_results_from_drive(paths):
    """Drive holds the record of earlier sessions: copy its result files locally."""
    d = paths.dirs
    for sub in (d["results"], d["smoke_results"]):
        src_dir = os.path.join(paths.drive, sub)
        for src in glob.glob(os.path.join(src_dir, "**", "*"), recursive=True):
            if os.path.isfile(src):
                dst = os.path.join(ROOT, sub, os.path.relpath(src, src_dir))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)


def push_result(paths, local, overwrite=True):
    rel = os.path.relpath(local, ROOT)
    copy_verified(local, os.path.join(paths.drive, rel), overwrite=overwrite)


# --------------------------------------------------------------------------- #
# inputs: checkpoints and D4 observations                                        #
# --------------------------------------------------------------------------- #


def ensure_checkpoint(cfg, table, task, seed):
    name = pv.checkpoint_name(task, seed)
    if name not in table:
        die(f"{name} is not in D6's table.")
    repo = cfg["source"]["checkpoints"].rstrip("/").split("huggingface.co/")[1]
    url = pv.HF_FILE.format(repo=repo, rev=cfg["source"]["hf_revision"], path=name)
    local = os.path.join(ROOT, cfg["paths"]["checkpoints"], name)
    try:
        return local, pv.download_verified(url, local, table[name], name)
    except pv.ProvenanceError as e:
        die(str(e))


def verify_d4(cfg, drive, needed):
    """The regenerated D4 observations: match: true, and each file as in the manifest."""
    d = cfg["planner_collect"]["d4"]
    meta = json.load(open(os.path.join(ROOT, d["meta"])))
    if meta.get("match") is not True:
        die(f"{d['meta']} does not record match: true; the D4 data needs a D7 deviation.")
    manifest = os.path.join(ROOT, d["manifest"])
    rows = {r["file"]: r for r in pv.read_manifest(manifest)}
    out = {}
    for task, seed in needed:
        f = f"{name_of(task, seed)}.npz"
        if f not in rows:
            die(f"{f} is not in {d['manifest']}.")
        path = os.path.join(drive, d["drive_dir"], f)
        if not os.path.exists(path):
            die(f"D4 observations {path} not found.")
        try:
            pv.verify_sha256(path, rows[f]["sha256"], f"D4 file {f}")
        except pv.ProvenanceError as e:
            die(str(e))
        out[(task, seed)] = path
    return out, dict(meta=d["meta"], meta_match=True, manifest=d["manifest"],
                     manifest_sha256=pv.sha256(manifest),
                     files={f"{name_of(*k)}.npz": rows[f"{name_of(*k)}.npz"]["sha256"] for k in out})


def d4_obs(path):
    O = np.load(path)["obs"]
    return np.asarray(O, np.float32).reshape(-1, O.shape[-1])


# --------------------------------------------------------------------------- #
# agents                                                                        #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Loaded:
    task: str
    seed: int
    layout: str
    input_name: str      # what the planner acts on: identity or symlog
    agent: object
    tcfg: object         # tdmpc2 config (dataclass)
    path: str
    sha256: str
    sd_np: dict          # original, unremapped weights, float64
    module_names: list
    first_layer: tuple   # (linear prefix, norm prefix) in sd_np
    key_mapping: dict = None


class TD:
    """tdmpc2's modules, imported once (after sys.path and cwd point at tdmpc2)."""

    def __init__(self, cfg):
        src = os.path.join(ROOT, cfg["paths"]["tdmpc2_src"])
        head = pv.git("rev-parse", "HEAD", cwd=src)[1]
        if head != cfg["source"]["repo_commit"]:
            die(f"tdmpc2 at {src} is {head!r}, expected {cfg['source']['repo_commit']}.")
        self.src, self.commit = src, head
        self.dir = os.path.join(src, "tdmpc2")
        sys.path.insert(0, self.dir)
        os.chdir(self.dir)
        import torch
        if not torch.cuda.is_available():
            die("a CUDA GPU is required (tdmpc2 hard-codes cuda:0).")
        torch.backends.cudnn.benchmark = True  # as evaluate.py
        from common.seed import set_seed
        from envs import make_env
        from tdmpc2 import TDMPC2
        self.torch, self.set_seed, self.make_env, self.TDMPC2 = torch, set_seed, make_env, TDMPC2


def load_agent(cfg, td, table, task, seed, res_dir):
    pc = cfg["planner_collect"]
    torch = td.torch
    path, sha = ensure_checkpoint(cfg, table, task, seed)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    mods = getattr(sd, "_metadata", None)
    sd_np = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in sd.items()
             if hasattr(v, "detach")}
    where = f"{task} seed {seed}"
    layout = layouts.detect_layout(sd_np, where)
    if layout != cfg["survey"][task][seed]:
        die(f"{where}: detected layout {layout}, survey (D5) says {cfg['survey'][task][seed]}.")
    first = layouts.check_first_layer(sd_np, layout, cfg["layer"]["layouts"], where)

    c = tdmpc2_cfg(td.dir, task, path, dict(episodes=pc["episodes_per_env_seed"],
                                           eval_seed=pc["env_seeds"][0], compile=pc["compile"]))
    td.make_env(c)  # sets obs_shape, action_dim, episode_length on c, as evaluate.py
    agent = td.TDMPC2(c)
    mapping = None
    if layout == layouts.PRERELEASE:
        new_sd, mapping = pl.remap_prerelease(dict(sd))
        target = {k: tuple(v.shape) for k, v in agent.model.state_dict().items()
                  if "Qs" not in k and not k.startswith("log_std")}
        got = {k: tuple(v.shape) for k, v in new_sd.items() if not k.startswith(pl.Q_PREFIXES)}
        spec = pl.public_shapes(c.obs_shape["state"][0], c.action_dim)
        if got != target or got != spec:
            die(f"{where}: remapped keys/shapes differ from the public WorldModel:\n"
                f"  missing {sorted(set(target) - set(got))}\n  extra {sorted(set(got) - set(target))}\n"
                f"  shape differences {[k for k in got if k in target and got[k] != target[k]]}")
        agent.load({"model": new_sd})  # api_model_conversion converts the Q keys
        write_json(os.path.join(res_dir, f"key_mapping_{name_of(task, seed)}.json"),
                   dict(task=task, seed=seed, checkpoint=pv.checkpoint_name(task, seed),
                        sha256=sha, d6_table=[row for row, _ in pl.PRERELEASE_REMAP],
                        mapping=mapping,
                        note="Q-ensemble keys pass through unchanged and are converted by "
                             "tdmpc2's own api_model_conversion inside TDMPC2.load."))
        input_name = "symlog"
    else:
        agent.load({"model": dict(sd)})
        input_name = "identity"
    print(f"loaded {where} [{layout}], planner input {input_name}, sha256 {sha[:12]}")
    return Loaded(task=task, seed=seed, layout=layout, input_name=input_name, agent=agent,
                  tcfg=c, path=path, sha256=sha, sd_np=sd_np,
                  module_names=None if mods is None else sorted(mods.keys()),
                  first_layer=first, key_mapping=mapping)


# --------------------------------------------------------------------------- #
# encoder agreement                                                             #
# --------------------------------------------------------------------------- #


def planner_encode(td, L, X32, batch):
    out = []
    with td.torch.no_grad():
        for i in range(0, len(X32), batch):
            t = td.torch.from_numpy(np.ascontiguousarray(X32[i:i + batch])).to(L.agent.device)
            out.append(L.agent.model.encode(t, None).double().cpu().numpy())
    return np.concatenate(out)


def agreement(cfg, td, L, planner_input, lens_input, d4_path, role):
    """Relative errors of the planner's float32 encoder (on the loaded object) against
    the lens code's float64 encoder on the original weights, on D6's two state sets."""
    g = cfg["planner_collect"]["gate"]
    eps0 = float(cfg["consistency"]["pos0_layernorm_eps"])
    eps, sdim = float(cfg["layer"]["layernorm_eps"]), int(cfg["consistency"]["simnorm_dim"])
    D4 = d4_obs(d4_path)
    sets = {"random": pl.random_states(D4, int(g["n_random"]), int(g["rng_seed"]),
                                       int(g["std_ddof"])).astype(np.float32),
            "d4": D4}
    enc = layouts.build_networks(L.sd_np, L.layout, eps, sdim, L.module_names,
                                 input_fn=pl.input_fn(lens_input, eps0))[0]
    lin = L.first_layer[0]
    E, b = L.sd_np[f"{lin}.weight"], L.sd_np[f"{lin}.bias"]
    d1 = pl.d1(pl.input_fn(lens_input, eps0)(D4.astype(np.float64)))
    errs, detail = {}, {}
    for s, X in sets.items():
        zp = planner_encode(td, L, pl.planner_input(X, planner_input, eps0), int(g["batch"]))
        zl = enc(X.astype(np.float64))
        errs[s] = pl.relative_errors(zp, zl)
        Xl = pl.input_fn(lens_input, eps0)(X.astype(np.float64))
        worst, info = pl.worst_states(errs[s], Xl, E, b, eps, d1, int(g["worst_n"]))
        detail[s] = dict(worst=worst, **info)
    res = pl.evaluate_sets(errs, g)
    for s in res["sets"]:
        res["sets"][s].update(detail[s])
    res.update(role=role, task=L.task, seed=L.seed, layout=L.layout,
               planner_input=planner_input, lens_input=lens_input, d4_file=os.path.basename(d4_path),
               std_ddof=int(g["std_ddof"]), d1_from="D4 policy-prior observations, lens-side input")
    print(f"  {role:9s} {L.task} seed {L.seed}: planner {planner_input}, lens {lens_input}: "
          + "; ".join(f"{s} median {r['median']:.2e} p99 {r['p99']:.2e} max {r['max']:.2e} "
                      f"{'pass' if r['pass'] else 'FAIL'}" for s, r in res["sets"].items()))
    return res


def gate_rows(stamp, results, smoke):
    rows = []
    for key, r in results.items():
        for s, x in r["sets"].items():
            rows.append(dict(session=stamp, kind="smoke" if smoke else "data", test=key, role=r["role"], task=r["task"], seed=r["seed"],
                             planner_input=r["planner_input"], lens_input=r["lens_input"], set=s,
                             n=x["n"], median=x["median"], p99=x["p99"], max=x["max"],
                             rule_pass=x["pass"],
                             worst_dist_over_r_eff_d1=";".join(f"{w['dist_over_r_eff_d1']:.4g}"
                                                               for w in x["worst"])))
    return rows


# --------------------------------------------------------------------------- #
# acting                                                                        #
# --------------------------------------------------------------------------- #


def act_episodes(cfg, td, L, env_seeds, episodes, steps):
    torch = td.torch
    eps0 = float(cfg["consistency"]["pos0_layernorm_eps"])
    c = L.tcfg
    O, A, R, S, I, T = [], [], [], [], [], []
    for s in env_seeds:
        c.seed = int(s)
        env = td.make_env(c)  # DMControl task_kwargs={'random': s}
        td.set_seed(int(s))
        for e in range(episodes):
            t0 = time.perf_counter()
            obs = env.reset()
            obs_l, act_l, rew_l = [obs.numpy().copy()], [], []
            done, t = False, 0
            while not done:
                inp = obs if L.input_name == "identity" else torch.from_numpy(
                    pl.planner_input(obs.numpy(), L.input_name, eps0))
                action = L.agent.act(inp, t0=t == 0, eval_mode=True)
                obs, reward, done, info = env.step(action)
                obs_l.append(obs.numpy().copy())
                act_l.append(action.numpy().copy())
                rew_l.append(float(reward))
                t += 1
            if t != steps:
                die(f"{L.task} seed {L.seed}: episode ran {t} steps, expected {steps}.")
            O.append(np.stack(obs_l)), A.append(np.stack(act_l)), R.append(np.array(rew_l, np.float32))
            S.append(s), I.append(e), T.append(time.perf_counter() - t0)
            print(f"    {L.task} seed {L.seed} env seed {s} episode {e}: return {sum(rew_l):.1f}, "
                  f"{T[-1]:.1f} s")
    return dict(obs=np.stack(O).astype(np.float32), actions=np.stack(A).astype(np.float32),
                rewards=np.stack(R), env_seed=np.array(S), episode_in_seed=np.array(I),
                seconds=np.array(T))


def versions(td):
    v = {}
    for p in PACKAGES:
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    v["torch_runtime"] = td.torch.__version__
    return v


def run_checkpoint(cfg, td, L, paths, session, smoke):
    pc = cfg["planner_collect"]
    stamp = session["stamp"]
    if smoke:
        seeds, eps, fname = [pc["smoke"]["env_seed"]], int(pc["smoke"]["episodes"]), \
            f"{name_of(L.task, L.seed)}-smoke-{stamp}"
    else:
        seeds, eps, fname = pc["env_seeds"], int(pc["episodes_per_env_seed"]), name_of(L.task, L.seed)
    drive_npz = os.path.join(paths.drive_data, f"{fname}.npz")
    if os.path.exists(drive_npz):  # left by an interrupted session (no complete meta)
        aside = f"{drive_npz}.incomplete-{stamp}"
        os.replace(drive_npz, aside)
        print(f"  moved an incomplete earlier data file aside: {aside}")
    t0 = time.time()
    data = act_episodes(cfg, td, L, seeds, eps, int(pc["steps"]))
    npz = os.path.join(paths.data, f"{fname}.npz")
    np.savez(npz, **data)
    sha = pv.sha256(npz)
    pub, pub_step = published_return(td.src, L.task, L.seed)
    rs = pl.return_summary(data["rewards"].astype(np.float64).sum(1), pub, float(pc["flag_below"]))
    meta = dict(stage="planner_collect", kind=paths.kind, complete=True,
                smoke=smoke, smoke_note=("outcome data for this checkpoint; never used for any "
                                         "rule and never counted toward the 50 episodes") if smoke else None,
                task=L.task, seed=L.seed, layout=L.layout, planner_input=L.input_name,
                eval_mode=True, env_seeds=[int(s) for s in seeds], episodes_per_env_seed=eps,
                steps=int(pc["steps"]), action_repeat=2,
                returns=rs, published_step=pub_step,
                published_source=f"tdmpc2 {td.commit} results/tdmpc2/{L.task}.csv, last logged step, seed {L.seed}",
                checkpoint=dict(file=pv.checkpoint_name(L.task, L.seed), sha256=L.sha256,
                                hf_revision=cfg["source"]["hf_revision"]),
                data=dict(file=os.path.relpath(npz, ROOT), drive=os.path.join(paths.drive_data, f"{fname}.npz"),
                          sha256=sha, bytes=os.path.getsize(npz),
                          shapes={k: list(v.shape) for k, v in data.items()}),
                key_mapping=(f"key_mapping_{name_of(L.task, L.seed)}.json" if L.key_mapping else None),
                encoder_gate=(session.get("gate_file") if L.layout == layouts.PRERELEASE else None),
                controls=(session.get("controls") if L.layout == layouts.PRERELEASE else None),
                gate_result=(session.get("gate", {}).get(name_of(L.task, L.seed))
                             if L.layout == layouts.PRERELEASE else None),
                seconds_total=time.time() - t0,
                seconds_per_episode=data["seconds"].tolist(),
                tdmpc2_config=dataclasses.asdict(L.tcfg), **session["env"],
                config=cfg, time=utc().isoformat(), session=stamp)
    meta_path = os.path.join(paths.res, f"meta_{fname}.json")
    write_json(meta_path, meta)
    copy_verified(npz, os.path.join(paths.drive_data, f"{fname}.npz"))
    push_result(paths, meta_path)
    if L.key_mapping:
        push_result(paths, os.path.join(paths.res, f"key_mapping_{name_of(L.task, L.seed)}.json"))
    print(f"  {L.task} seed {L.seed}: mean return {rs['return_mean']:.1f} over {rs['episodes']} "
          f"episodes, published {pub} (fraction {rs['fraction']:.3f})"
          + ("  FLAG < 0.5" if rs["flag_below_half"] else "") + ("  [smoke]" if smoke else ""))
    return meta


def is_complete(paths, task, seed, cfg):
    meta = os.path.join(paths.res, f"meta_{name_of(task, seed)}.json")
    npz = os.path.join(paths.drive_data, f"{name_of(task, seed)}.npz")
    if not (os.path.exists(meta) and os.path.exists(npz)):
        return False
    m = json.load(open(meta))
    pc = cfg["planner_collect"]
    want = len(pc["env_seeds"]) * int(pc["episodes_per_env_seed"])
    return (m.get("complete") is True and not m.get("smoke") and m["returns"]["episodes"] == want
            and pv.sha256(npz) == m["data"]["sha256"])


# --------------------------------------------------------------------------- #
# summaries                                                                     #
# --------------------------------------------------------------------------- #


def write_summaries(cfg, paths):
    d = cfg["planner_collect"]["dirs"]
    metas = []
    for sub in (d["results"], d["smoke_results"]):
        for p in sorted(glob.glob(os.path.join(ROOT, sub, "meta_*.json"))):
            m = json.load(open(p))
            if m.get("stage") == "planner_collect" and m.get("complete"):
                metas.append(m)
    ret_rows, man_rows = [], []
    for m in metas:
        r = m["returns"]
        ret_rows.append(dict(kind=m["kind"], task=m["task"], seed=m["seed"], layout=m["layout"],
                             planner_input=m["planner_input"], eval_mode=m["eval_mode"],
                             episodes=r["episodes"], return_mean=r["return_mean"],
                             return_std=r["return_std"], return_min=r["return_min"],
                             return_max=r["return_max"], published=r["published"],
                             published_step=m["published_step"], fraction=r["fraction"],
                             flag_below_half=r["flag_below_half"], gpu=m.get("gpu"),
                             session=m["session"], data_file=os.path.basename(m["data"]["file"])))
        man_rows.append(dict(kind=m["kind"], file=os.path.relpath(m["data"]["drive"], paths.drive),
                             bytes=m["data"]["bytes"], sha256=m["data"]["sha256"]))
    out = []
    if ret_rows:
        p = os.path.join(paths.summary, "planner_returns.csv")
        write_csv(p, ret_rows)
        out.append(p)
        p = os.path.join(paths.summary, "planner_obs_manifest.csv")
        write_csv(p, man_rows, ["kind", "file", "bytes", "sha256"])
        out.append(p)
    rows = []
    for sub in (d["results"], d["smoke_results"]):
        for p in sorted(glob.glob(os.path.join(ROOT, sub, "gate", "gate_*.json"))):
            g = json.load(open(p))
            rows += gate_rows(g["session"], g["tests"], g["smoke"])
    if rows:
        p = os.path.join(paths.summary, "encoder_gate.csv")
        write_csv(p, rows)
        out.append(p)
    for p in out:
        push_result(paths, p)


def final_summary(cfg, paths, session, status):
    d = cfg["planner_collect"]["dirs"]
    files = []
    for sub in (d["results"], d["smoke_results"]):
        files += [p for p in glob.glob(os.path.join(ROOT, sub, "**", "*"), recursive=True)
                  if os.path.isfile(p)]
    print("\n================ summary ================")
    print(f"session {session['stamp']}: {status}")
    for r in session["halts"]["reasons"]:
        print(f"  halt: {r}")
    for k, v in session["checkpoints"].items():
        print(f"  {k:24s} {v}")
    smoke = sorted(glob.glob(os.path.join(ROOT, d["smoke_results"], "meta_*.json")))
    if smoke:
        print("smoke outputs (outcome data; never used for any rule):")
        for p in smoke:
            m = json.load(open(p))
            print(f"  {m['data']['drive']}  sha256 {m['data']['sha256'][:12]}  "
                  f"return {m['returns']['return_mean']:.1f}")
    big = [p for p in files if os.path.getsize(p) >= 1 << 20]
    if big:
        print(f"WARNING: over 1 MB, do not commit as is: {[os.path.relpath(p, ROOT) for p in big]}")
    print("\nfiles to commit:")
    print(pv.git_add_command([p for p in files if p not in big]))


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--drive", required=True, help="the Drive folder MyDrive/layernorm-lens-r6")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    try:
        pv.require_prereg()
        table = pv.d6_sha_table()
    except pv.ProvenanceError as e:
        die(str(e))
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if not os.path.isdir(args.drive):
        die(f"--drive {args.drive!r} is not a directory (mount Google Drive first).")
    pc = cfg["planner_collect"]
    if pc["eval_mode"] is not True:
        die("D6 (a) fixes eval_mode=True; planner_collect.eval_mode must be true.")
    paths = Paths(cfg, os.path.abspath(args.drive), args.smoke)
    pull_results_from_drive(paths)
    stamp = utc().strftime("%Y%m%dT%H%M%SZ")

    pos = (pc["positive_control"]["task"], int(pc["positive_control"]["seed"]))
    neg = (pc["negative_control"]["task"], int(pc["negative_control"]["seed"]))
    pre = [(p["task"], int(p["seed"])) for p in pc["prerelease"]]
    public = [(t, int(s)) for t in cfg["tasks"] for s in cfg["seeds"]
              if cfg["survey"][t][s] == layouts.PUBLIC]
    if args.smoke:
        todo_public, todo_pre = [], [(pc["smoke"]["task"], int(pc["smoke"]["seed"]))]
    else:
        todo_public = [k for k in public if not is_complete(paths, *k, cfg)]
        todo_pre = [k for k in pre if not is_complete(paths, *k, cfg)]
    session = dict(stamp=stamp, smoke=args.smoke, checkpoints={}, halts=dict(reasons=[]))
    for k in public + pre:
        if k not in todo_public + todo_pre and not args.smoke:
            session["checkpoints"][name_of(*k)] = "complete (earlier session)"
    if not todo_public and not todo_pre:
        print("every checkpoint is complete on Drive; nothing to run.")
        write_summaries(cfg, paths)
        final_summary(cfg, paths, session, "nothing to run")
        return

    td = TD(cfg)
    import torch
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        print(f"NOTE: GPU is {gpu}, not a Tesla T4 (D6 (e)); recorded.")
    session["env"] = dict(**pv.git_state(), prereg_tags=pv.tag_objects(), tdmpc2_commit=td.commit,
                          gpu=gpu, cuda=torch.version.cuda, python=platform.python_version(),
                          versions=versions(td), jax="not used (tdmpc2 environment; our float64 "
                                                     "work is numpy)")
    session["d6_table_source"] = f"git show {pv.D6_TAG}:{pv.D6_PATH}"
    d4_paths, session["d4"] = verify_d4(cfg, paths.drive, [pos, neg] + [k for k in pre])

    # controls, then the gate, on the objects that later act
    loaded = {}
    for k in [pos, neg] + [k for k in pre if k != neg]:
        loaded[k] = load_agent(cfg, td, table, *k, paths.res)
    tests = {}
    print("\ncontrols:")
    tests["positive_control"] = agreement(cfg, td, loaded[pos], "identity", "identity",
                                          d4_paths[pos], "positive")
    tests["negative_control"] = agreement(cfg, td, loaded[neg], "identity", "symlog",
                                          d4_paths[neg], "negative")
    controls = pl.control_outcomes(tests["positive_control"], tests["negative_control"],
                                   float(pc["gate"]["negative_min_max"]))
    gate = None
    if controls["as_stated"]:
        print("gate:")
        gate = {}
        for k in pre:
            tests[f"gate_{name_of(*k)}"] = gate[name_of(*k)] = agreement(
                cfg, td, loaded[k], "symlog", "symlog", d4_paths[k], "gate")
    session["controls"], session["gate"] = controls, gate
    session["halts"] = pl.halts(controls, gate)
    gate_file = os.path.join(paths.res, "gate", f"gate_{stamp}.json")
    write_json(gate_file, dict(session=stamp, smoke=args.smoke, tests=tests, controls=controls,
                               halts=session["halts"], rule=pc["gate"], **session["env"]))
    session["gate_file"] = os.path.relpath(gate_file, ROOT)
    push_result(paths, gate_file)
    print(f"controls as stated: {controls['as_stated']}; gate: "
          f"{'not run' if gate is None else {k: v['pass'] for k, v in gate.items()}}")

    status = "ok"
    if not session["halts"]["public_allowed"]:
        for k in todo_public + todo_pre:
            session["checkpoints"][name_of(*k)] = "halted: " + "; ".join(session["halts"]["reasons"])
        status = "HALTED (control)"
    else:
        for k in todo_public:
            L = loaded.pop(k, None) or load_agent(cfg, td, table, *k, paths.res)
            print(f"\n=== {name_of(*k)} ===")
            run_checkpoint(cfg, td, L, paths, session, smoke=False)
            session["checkpoints"][name_of(*k)] = "collected this session"
            write_summaries(cfg, paths)
            del L
            torch.cuda.empty_cache()
        for k in todo_pre:
            if not session["halts"]["prerelease_allowed"]:
                session["checkpoints"][name_of(*k)] = "halted: " + "; ".join(session["halts"]["reasons"])
                status = "HALTED (gate): pre-release runs not made"
                continue
            print(f"\n=== {name_of(*k)}{' (smoke)' if args.smoke else ''} ===")
            run_checkpoint(cfg, td, loaded[k], paths, session, smoke=args.smoke)
            session["checkpoints"][name_of(*k)] = ("smoke episode collected" if args.smoke
                                                   else "collected this session")
            write_summaries(cfg, paths)
    sess_path = os.path.join(paths.res, "sessions", f"session_{stamp}.json")
    write_json(sess_path, dict(session, config=cfg, time_end=utc().isoformat(), status=status))
    push_result(paths, sess_path)
    write_summaries(cfg, paths)
    final_summary(cfg, paths, session, status)
    if status.startswith("HALTED (control)"):
        sys.exit(EXIT_CONTROL)
    if status.startswith("HALTED (gate)"):
        sys.exit(EXIT_GATE)


if __name__ == "__main__":
    main()
