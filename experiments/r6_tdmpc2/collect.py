"""R6 data collection (D4) and the position-0 latent-consistency test (D3).

    python experiments/r6_tdmpc2/collect.py --config experiments/r6_tdmpc2/config.yaml collect
    python experiments/r6_tdmpc2/collect.py --config experiments/r6_tdmpc2/config.yaml consistency

Refuses to run unless the annotated tags prereg-r6, prereg-r6-d1 and prereg-r6-d2
exist and prereg/r6.md, r6-deviations.md and r6-deviations-2.md match them.

Nothing here computes a criterion quantity (Inside, Populated, Sharp) or the gate.

collect      For each task and seed: load the released checkpoint (downloaded by
             extract.py), act with the policy prior alone, a_t = tanh(mu(enc(o_t))),
             with identity at the encoder's input (D4, D5), in stock DMControl with the
             public TD-MPC2 conventions: suite.load(domain, task,
             task_kwargs={'random': env_seed}), actions scaled to [-1, 1],
             observation dict flattened in spec order as float32, action repeat 2,
             500 agent steps. 10 episodes per environment seed 0-4. Saves every
             observation (501 per episode, reset included), action and reward to
             data_out, and returns.csv (policy-only returns vs the published
             return; flag below 50%, no gate).
consistency  D3 on the pre-release checkpoints (config consistency.targets: the
             gate checkpoint and the reported one), and the same computation on
             the public-layout calibration checkpoints as a reference (D5). Writes
             consistency.csv / consistency.json; exits with status 2 and prints
             STOP if identity is rejected where consistency.stop_on_reject says so.

Each checkpoint's layout is detected from its keys (layouts.py, D5) and must agree
with the config's survey. Networks are evaluated in float64 numpy from the float32
checkpoint weights, following the public code for both layouts: Mish after every
LayerNorm except a network's last, SimNorm (simnorm_dim) after the last LayerNorm of
the encoder and the dynamics model, nothing after the policy's final Linear.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import platform
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layouts  # noqa: E402
TAGS = [("prereg-r6", "prereg/r6.md"), ("prereg-r6-d1", "prereg/r6-deviations.md"),
        ("prereg-r6-d2", "prereg/r6-deviations-2.md")]


# --------------------------------------------------------------------------- #
# provenance                                                                    #
# --------------------------------------------------------------------------- #


def git(*args, cwd=ROOT):
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout.strip()


def require_prereg():
    for tag, path in TAGS:
        rc, kind = git("cat-file", "-t", f"refs/tags/{tag}")
        if rc != 0:
            sys.exit(f"refusing to run: git tag {tag} does not exist (CLAUDE.md).")
        if kind != "tag":
            sys.exit(f"refusing to run: {tag} is a lightweight tag; an annotated tag is required.")
        rc, _ = git("diff", "--quiet", tag, "--", path)
        if rc != 0:
            sys.exit(f"refusing to run: {path} differs from the version tagged {tag}.")


def write_meta(cfg, stage, extra):
    import torch
    _, commit = git("rev-parse", "HEAD")
    _, dirty = git("status", "--porcelain", "--untracked-files=no")
    m = dict(stage=stage, git_commit=commit, git_dirty=bool(dirty),
             prereg_tags={t: git("rev-parse", t)[1] for t, _ in TAGS},
             jax=jax.__version__, numpy=np.__version__, torch=torch.__version__,
             python=platform.python_version(),
             time=datetime.datetime.now(datetime.timezone.utc).isoformat(), config=cfg)
    try:
        import dm_control
        import mujoco
        m.update(dm_control=getattr(dm_control, "__version__", "unknown"),
                 mujoco=mujoco.__version__)
    except ImportError:
        pass
    m.update(extra)
    out = os.path.join(ROOT, cfg["paths"]["results_out"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"meta_{stage}.json"), "w") as f:
        json.dump(m, f, indent=2)


# --------------------------------------------------------------------------- #
# networks from the released state_dict                                         #
# --------------------------------------------------------------------------- #


def load_checkpoint(cfg, task, seed):
    """float64 numpy state_dict and the module-name metadata of one checkpoint."""
    import torch
    ck_dir = os.path.join(ROOT, cfg["paths"]["checkpoints"])
    hits = [p for p in glob.glob(os.path.join(ck_dir, "**", "*.pt"), recursive=True)
            if os.path.basename(p) == f"{task}-{seed}.pt"]
    if len(hits) != 1:
        sys.exit(f"{task} seed {seed}: expected one checkpoint under {ck_dir}, found {hits}. "
                 "Run extract.py extract first.")
    ck = torch.load(hits[0], map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    mods = getattr(sd, "_metadata", None)
    arrays = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in sd.items()
              if hasattr(v, "detach")}
    return arrays, (None if mods is None else sorted(mods.keys())), hits[0]


def agent(cfg, task, seed, cand="identity"):
    """Encoder, dynamics and policy of one checkpoint, read according to its detected
    layout (D5), with D3 candidate `cand` at the encoder's input."""
    sd, mods, path = load_checkpoint(cfg, task, seed)
    where = f"{task} seed {seed}"
    layout = layouts.detect_layout(sd, where)
    expected = cfg["survey"][task][seed]
    if layout != expected:
        sys.exit(f"{where}: detected layout {layout}, survey (D5) says {expected}.")
    layouts.check_first_layer(sd, layout, cfg["layer"]["layouts"], where)
    c = cfg["consistency"]
    eps, sdim = float(cfg["layer"]["layernorm_eps"]), int(c["simnorm_dim"])
    inp = layouts.input_candidates(float(c["pos0_layernorm_eps"]))[cand]
    enc, dyn, pi = layouts.build_networks(sd, layout, eps, sdim, mods, input_fn=inp)
    return dict(enc=enc, dyn=dyn, pi=pi, layout=layout, path=path, modules=mods)


# --------------------------------------------------------------------------- #
# environment (public TD-MPC2 DMControl conventions)                            #
# --------------------------------------------------------------------------- #


def make_env(task, env_seed):
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale
    domain, name = task.replace("-", "_").split("_", 1)
    domain = dict(cup="ball_in_cup", pointmass="point_mass").get(domain, domain)
    env = suite.load(domain, name, task_kwargs={"random": env_seed}, visualize_reward=False)
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


def flat(obs):
    return np.concatenate([np.asarray(v).flatten() for v in obs.values()], dtype=np.float32)


def run_episode(env, act, steps, repeat):
    ts = env.reset()
    obs, acts, rews = [flat(ts.observation)], [], []
    dtype = env.action_spec().dtype
    for _ in range(steps):
        a = act(obs[-1])
        r = 0.0
        for _ in range(repeat):
            ts = env.step(a.astype(dtype))
            r += ts.reward or 0.0
        obs.append(flat(ts.observation))
        acts.append(a)
        rews.append(r)
    return np.stack(obs), np.stack(acts), np.array(rews)


def published_return(cfg, task, seed):
    """Reward at the last step logged for this seed in tdmpc2 results/tdmpc2/<task>.csv."""
    path = os.path.join(ROOT, cfg["paths"]["tdmpc2_src"], "results", "tdmpc2", f"{task}.csv")
    if not os.path.exists(path):
        return np.nan, None
    rows = [r for r in csv.DictReader(open(path)) if int(r["seed"]) == seed]
    if not rows:
        return np.nan, None
    last = max(rows, key=lambda r: int(r["step"]))
    return float(last["reward"]), int(last["step"])


def stage_collect(cfg):
    d = cfg["data"]
    out = os.path.join(ROOT, cfg["paths"]["data_out"])
    os.makedirs(out, exist_ok=True)
    rows, record = [], {}
    for task in cfg["tasks"]:
        for seed in cfg["seeds"]:
            ag = agent(cfg, task, seed, cand="identity")
            enc, pi, dyn = ag["enc"], ag["pi"], ag["dyn"]
            n_act = int(np.prod(make_env(task, 0).action_spec().shape))
            lat = enc.ops[-1][2][0]
            if pi.ops[-1][2][0] != 2 * n_act or dyn.ops[0][2][1] != lat + n_act:
                sys.exit(f"{task}: network sizes do not fit the environment: action dim "
                         f"{n_act}, latent {lat}, pi out {pi.ops[-1][2][0]}, "
                         f"dynamics in {dyn.ops[0][2][1]}")

            def act(o):
                mu, _ = np.split(pi(enc(o)), 2, axis=-1)
                return np.tanh(mu)

            O, A, R, S = [], [], [], []
            for es in d["env_seeds"]:
                env = make_env(task, es)
                for _ in range(d["episodes_per_task"] // len(d["env_seeds"])):
                    o, a, r = run_episode(env, act, d["steps"], d["action_repeat"])
                    O.append(o), A.append(a), R.append(r), S.append(es)
            O, A, R = np.stack(O), np.stack(A), np.stack(R)
            np.savez(os.path.join(out, f"{task}-seed{seed}.npz"), obs=O, actions=A,
                     rewards=R, env_seed=np.array(S), checkpoint=ag["path"])
            ret = R.sum(1)
            pub, pub_step = published_return(cfg, task, seed)
            frac = ret.mean() / pub if np.isfinite(pub) and pub else np.nan
            rows.append(dict(task=task, seed=seed, layout=ag["layout"], k=O.shape[-1],
                             episodes=len(ret),
                             return_mean=ret.mean(), return_std=ret.std(ddof=1),
                             published=pub, published_step=pub_step, fraction=frac,
                             flag_below_half=bool(np.isfinite(frac) and frac < 0.5)))
            record[f"{task}/{seed}"] = dict(checkpoint=ag["path"], layout=ag["layout"],
                                            enc=ag["enc"].ops,
                                            dyn=ag["dyn"].ops, pi=ag["pi"].ops,
                                            parameter_free={n: ag[n].implied_free
                                                            for n in ("enc", "dyn", "pi")})
            print(f"{task} seed {seed} [{ag['layout']}]: k = {O.shape[-1]}, return {ret.mean():.1f} "
                  f"(published {pub}, fraction {frac:.2f}){'  FLAG < 50%' if rows[-1]['flag_below_half'] else ''}")
    _csv(os.path.join(ROOT, cfg["paths"]["results_out"], "returns.csv"), rows)
    write_meta(cfg, "collect", dict(networks=record))


# --------------------------------------------------------------------------- #
# D3 latent-consistency test                                                    #
# --------------------------------------------------------------------------- #


def consistency_errors(ag_by_cand, O, A, rng_seed):
    """e_c and e0_c for each candidate on one task's transitions."""
    o_t = O[:, :-1].reshape(-1, O.shape[-1])
    o_n = O[:, 1:].reshape(-1, O.shape[-1])
    a_t = A.reshape(-1, A.shape[-1])
    n = len(o_t)
    j = np.random.default_rng(rng_seed).integers(0, n, size=n)
    out = {}
    for c, ag in ag_by_cand.items():
        z_t, z_n = ag["enc"](o_t), ag["enc"](o_n)
        pred = ag["dyn"](np.concatenate([z_t, a_t], -1))
        out[c] = dict(e=float(((pred - z_n) ** 2).sum(-1).mean()),
                      e0=float(((pred - z_n[j]) ** 2).sum(-1).mean()), n=n)
    return out


def decide(err, accept_ratio, tie_ratio):
    """D3: identity accepted iff lowest e and e_identity <= accept_ratio * e0_identity;
    candidates with e <= tie_ratio * e_identity are reported alongside."""
    e_id = err["identity"]["e"]
    lowest = min(err, key=lambda c: err[c]["e"])
    identity_lowest = all(e_id <= err[c]["e"] for c in err)
    accepted = identity_lowest and e_id <= accept_ratio * err["identity"]["e0"]
    dual = [c for c in err if c != "identity" and err[c]["e"] <= tie_ratio * e_id]
    return dict(lowest=lowest, identity_lowest=identity_lowest, accepted=accepted,
                report_under_both=dual)


def stage_consistency(cfg):
    """D3 on the pre-release targets (D5); the same computation on the public-layout
    calibration checkpoints, reported as a reference only."""
    c = cfg["consistency"]
    jobs = ([(t["task"], int(t["seed"]), t["role"], layouts.PRERELEASE) for t in c["targets"]]
            + [(t["task"], int(t["seed"]), "calibration", layouts.PUBLIC) for t in c["calibration"]])
    rows, verdicts, stop = [], {}, []
    for task, seed, role, want in jobs:
        data = np.load(os.path.join(ROOT, cfg["paths"]["data_out"], f"{task}-seed{seed}.npz"))
        ags = {cand: agent(cfg, task, seed, cand=cand) for cand in c["candidates"]}
        layout = ags["identity"]["layout"]
        if layout != want:
            sys.exit(f"{task} seed {seed}: {role} requires the {want} layout, detected {layout}.")
        err = consistency_errors(ags, data["obs"], data["actions"], int(c["rng_seed"]))
        v = decide(err, float(c["accept_ratio"]), float(c["tie_ratio"]))
        if role == "calibration":  # reference only: no decision is applied
            v = dict(lowest=v["lowest"], accepted=None, report_under_both=[])
        # D5: a rejected checkpoint that does not stop the analysis is unidentified,
        # and its R6 results are reported under every candidate
        v["identified"] = None if v["accepted"] is None else bool(v["accepted"])
        v["report_r6_under"] = (["identity"] + v["report_under_both"] if v["accepted"]
                                else list(err) if v["accepted"] is False else [])
        verdicts[f"{task}/{seed}"] = dict(role=role, layout=layout, errors=err, **v)
        for cand, x in err.items():
            rows.append(dict(task=task, seed=seed, role=role, layout=layout, candidate=cand,
                             e=x["e"], e0=x["e0"], e_over_e0=x["e"] / x["e0"], n=x["n"],
                             lowest=(cand == v["lowest"]),
                             identity_accepted="" if v["accepted"] is None else v["accepted"],
                             identified="" if v["identified"] is None else v["identified"],
                             report_under_both=";".join(v["report_under_both"]),
                             report_r6_under=";".join(v["report_r6_under"])))
        verdict = ("reference only" if role == "calibration"
                   else "identity ACCEPTED" if v["accepted"]
                   else "identity REJECTED" + ("" if c["stop_on_reject"][role]
                                               else ": checkpoint UNIDENTIFIED, R6 under each candidate"))
        print(f"{task} seed {seed} [{role}, {layout}]: "
              + ", ".join(f"{k} e/e0={x['e'] / x['e0']:.4g}" for k, x in err.items())
              + f"  -> {verdict}"
              + (f"; report under both with {v['report_under_both']}" if v["report_under_both"] else ""))
        if role != "calibration" and not v["accepted"] and c["stop_on_reject"][role]:
            stop.append(f"{task} seed {seed} ({role})")
    res = os.path.join(ROOT, cfg["paths"]["results_out"])
    _csv(os.path.join(res, "consistency.csv"), rows)
    with open(os.path.join(res, "consistency.json"), "w") as f:
        json.dump(verdicts, f, indent=2)
    write_meta(cfg, "consistency", {})
    if stop:
        print(f"\nSTOP: identity rejected for {stop}. Consult the author (D3, D5).")
        sys.exit(2)


def _csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("stage", choices=["collect", "consistency"])
    args = ap.parse_args()
    require_prereg()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dict(collect=stage_collect, consistency=stage_consistency)[args.stage](cfg)


if __name__ == "__main__":
    main()
