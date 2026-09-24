"""R6 data collection (D4) and the position-0 latent-consistency test (D3).

    python experiments/r6_tdmpc2/collect.py --config experiments/r6_tdmpc2/config.yaml collect
    python experiments/r6_tdmpc2/collect.py --config experiments/r6_tdmpc2/config.yaml consistency

Refuses to run unless the annotated tags prereg-r6 and prereg-r6-d1 exist and
prereg/r6.md and prereg/r6-deviations.md match their tagged versions.

Nothing here computes a criterion quantity (Inside, Populated, Sharp) or the gate.

collect      For each task and seed: load the released checkpoint (downloaded by
             extract.py), act with the policy prior alone, a_t = tanh(mu(enc(o_t))),
             with encoder position 0 = identity (D4), in stock DMControl with the
             public TD-MPC2 conventions: suite.load(domain, task,
             task_kwargs={'random': env_seed}), actions scaled to [-1, 1],
             observation dict flattened in spec order as float32, action repeat 2,
             500 agent steps. 10 episodes per environment seed 0-4. Saves every
             observation (501 per episode, reset included), action and reward to
             data_out, and returns.csv (policy-only returns vs the published
             return; flag below 50%, no gate).
consistency  Seed 1 per task: e_c for each position-0 candidate and e0, and the D3
             decision. Writes consistency.csv / consistency.json; exits with status
             2 and prints STOP if identity is rejected for any task.

Networks are evaluated in float64 numpy from the float32 checkpoint weights.
Parameter-free positions follow the public code (D3): Mish after every LayerNorm
except a network's last, SimNorm (simnorm_dim) after the last LayerNorm of the
encoder and the dynamics model, nothing after the policy's final Linear.
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
TAGS = [("prereg-r6", "prereg/r6.md"), ("prereg-r6-d1", "prereg/r6-deviations.md")]


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


def _path_key(p):
    return tuple(int(x) for x in p.split("."))


def build_net(sd, prefix, final, eps, simnorm_dim, module_names=None, pos0=None):
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
        ops.append(("linear" if w.ndim == 2 else "ln", p, w, b))
    # parameter-free positions implied by the rule
    implied = []
    for i, (kind, p, _, _) in enumerate(ops):
        if kind == "ln":
            parent, idx = (p.rsplit(".", 1) if "." in p else ("", p))
            nxt = f"{parent}.{int(idx) + 1}" if parent else str(int(idx) + 1)
            implied.append(nxt)
    if pos0 is not None:
        implied.append("0")
    present = {p for _, p, _, _ in ops}
    for q in implied:
        if q in present:
            errs.append(f"{prefix}.{q}: rule implies a parameter-free module, found parameters")
    # gaps within each Sequential level must be exactly the implied positions
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
        sys.exit("layout check failed:\n  " + "\n  ".join(errs))

    def f(x):
        x = np.asarray(x, np.float64)
        if pos0 is not None:
            x = pos0(x)
        for i, (kind, p, w, b) in enumerate(ops):
            if kind == "linear":
                x = x @ w.T + b
            else:
                m = x.mean(-1, keepdims=True)
                v = ((x - m) ** 2).mean(-1, keepdims=True)
                x = (x - m) / np.sqrt(v + eps) * w + b
                x = simnorm(x, simnorm_dim) if (i == len(ops) - 1 and final == "simnorm") else mish(x)
        return x

    f.ops = [(k, p, tuple(w.shape)) for k, p, w, _ in ops]
    f.implied_free = sorted(set(implied), key=_path_key)
    return f


def mish(x):
    return x * np.tanh(np.logaddexp(0.0, x))


def simnorm(x, dim):
    s = x.shape
    x = x.reshape(*s[:-1], -1, dim)
    x = np.exp(x - x.max(-1, keepdims=True))
    return (x / x.sum(-1, keepdims=True)).reshape(s)


def pos0_candidates(eps):
    def layernorm(o):
        m = o.mean(-1, keepdims=True)
        return (o - m) / np.sqrt(((o - m) ** 2).mean(-1, keepdims=True) + eps)
    return {"identity": lambda o: o,
            "symlog": lambda o: np.sign(o) * np.log1p(np.abs(o)),
            "layernorm": layernorm}


def agent(cfg, task, seed, pos0="identity"):
    sd, mods, path = load_checkpoint(cfg, task, seed)
    c = cfg["consistency"]
    eps, sdim = float(cfg["layer"]["layernorm_eps"]), int(c["simnorm_dim"])
    enc = build_net(sd, "_encoder.state", "simnorm", eps, sdim, mods,
                    pos0=pos0_candidates(float(c["pos0_layernorm_eps"]))[pos0])
    dyn = build_net(sd, "_dynamics", "simnorm", eps, sdim, mods)
    pi = build_net(sd, "_pi", None, eps, sdim, mods)
    return dict(enc=enc, dyn=dyn, pi=pi, path=path, sd_keys=sorted(sd), modules=mods)


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
            ag = agent(cfg, task, seed, pos0="identity")
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
            rows.append(dict(task=task, seed=seed, k=O.shape[-1], episodes=len(ret),
                             return_mean=ret.mean(), return_std=ret.std(ddof=1),
                             published=pub, published_step=pub_step, fraction=frac,
                             flag_below_half=bool(np.isfinite(frac) and frac < 0.5)))
            record[f"{task}/{seed}"] = dict(checkpoint=ag["path"], enc=ag["enc"].ops,
                                            dyn=ag["dyn"].ops, pi=ag["pi"].ops,
                                            parameter_free={n: ag[n].implied_free
                                                            for n in ("enc", "dyn", "pi")})
            print(f"{task} seed {seed}: k = {O.shape[-1]}, return {ret.mean():.1f} "
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
    c = cfg["consistency"]
    seed = int(c["seed"])
    rows, verdicts = [], {}
    for task in cfg["tasks"]:
        data = np.load(os.path.join(ROOT, cfg["paths"]["data_out"], f"{task}-seed{seed}.npz"))
        ags = {cand: agent(cfg, task, seed, pos0=cand) for cand in c["candidates"]}
        err = consistency_errors(ags, data["obs"], data["actions"], int(c["rng_seed"]))
        v = decide(err, float(c["accept_ratio"]), float(c["tie_ratio"]))
        verdicts[task] = dict(errors=err, **v)
        for cand, x in err.items():
            rows.append(dict(task=task, seed=seed, candidate=cand, e=x["e"], e0=x["e0"],
                             e_over_e0=x["e"] / x["e0"], n=x["n"],
                             lowest=(cand == v["lowest"]), identity_accepted=v["accepted"],
                             report_under_both=";".join(v["report_under_both"])))
        print(f"{task}: " + ", ".join(f"{k} e={x['e']:.4g} (e0 {x['e0']:.4g})" for k, x in err.items())
              + f"  -> identity {'ACCEPTED' if v['accepted'] else 'REJECTED'}"
              + (f"; report under both with {v['report_under_both']}" if v["report_under_both"] else ""))
    res = os.path.join(ROOT, cfg["paths"]["results_out"])
    _csv(os.path.join(res, "consistency.csv"), rows)
    with open(os.path.join(res, "consistency.json"), "w") as f:
        json.dump(verdicts, f, indent=2)
    write_meta(cfg, "consistency", {})
    rejected = [t for t, v in verdicts.items() if not v["accepted"]]
    if rejected:
        print(f"\nSTOP: identity rejected for {rejected}. Consult the author (D3).")
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
