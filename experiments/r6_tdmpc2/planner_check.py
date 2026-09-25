"""R6 step-1 feasibility: evaluate public-layout checkpoints WITH planning, through
the public tdmpc2 code's own TDMPC2 class (tdmpc2 e9f59321), on a GPU.

    python experiments/r6_tdmpc2/planner_check.py --config experiments/r6_tdmpc2/config.yaml

Runs in its own Python 3.11 environment with tdmpc2's pinned dependencies
(docker/environment.yaml), not the project's JAX environment: this entry point runs
the authors' float32 PyTorch agent and imports neither JAX nor lens/. It computes no
lens, coverage or criterion quantity; it reports episode returns against the
published ones and wall-clock time per episode.

For each checkpoint in config `planner_check.checkpoints`:
  - refuse unless the annotated tags prereg-r6, -d1, -d2 exist and the files match;
  - require tdmpc2 at source.repo_commit, and a public-layout checkpoint (layouts.py;
    pre-release checkpoints cannot be loaded by the public code);
  - build the config exactly as evaluate.py does (config.yaml + overrides, parse_cfg,
    make_env, set_seed), load the state dict with torch.load(weights_only=True) and
    pass it to TDMPC2.load(), which applies api_model_conversion;
  - run the episodes of evaluate.py's loop, timing each, once per eval_mode in
    config `planner_check.eval_modes`: evaluate.py calls agent.act(obs, t0=t==0)
    (eval_mode False: the planner's action plus exploration noise, tdmpc2.py:203),
    while the training-time evaluations behind the published curves call
    agent.act(obs, t0=t==0, eval_mode=True) (trainer/online_trainer.py:37).
Writes planner_check.csv and meta_planner_check.json to results_out.

parse_cfg() calls hydra.utils.get_original_cwd(), which needs a running Hydra app and
only sets the log directory (work_dir); it is replaced by os.getcwd here.
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
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import layouts  # noqa: E402  (numpy only)

TAGS = [("prereg-r6", "prereg/r6.md"), ("prereg-r6-d1", "prereg/r6-deviations.md"),
        ("prereg-r6-d2", "prereg/r6-deviations-2.md")]


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
        if git("diff", "--quiet", tag, "--", path)[0] != 0:
            sys.exit(f"refusing to run: {path} differs from the version tagged {tag}.")


def published_return(src, task, seed):
    """Reward at the last step logged for this seed in tdmpc2 results/tdmpc2/<task>.csv."""
    path = os.path.join(src, "results", "tdmpc2", f"{task}.csv")
    rows = [r for r in csv.DictReader(open(path)) if int(r["seed"]) == seed]
    last = max(rows, key=lambda r: int(r["step"]))
    return float(last["reward"]), int(last["step"])


def find_checkpoint(cfg, task, seed):
    ck_dir = os.path.join(ROOT, cfg["paths"]["checkpoints"])
    hits = [p for p in glob.glob(os.path.join(ck_dir, "**", "*.pt"), recursive=True)
            if os.path.basename(p) == f"{task}-{seed}.pt"]
    if len(hits) != 1:
        sys.exit(f"{task} seed {seed}: expected one checkpoint under {ck_dir}, found {hits}. "
                 "Run extract.py extract first.")
    return hits[0]


def tdmpc2_cfg(tdmpc2_dir, task, checkpoint, pc):
    """evaluate.py's config: config.yaml with overrides, then parse_cfg."""
    import hydra.utils
    from omegaconf import OmegaConf
    from common.parser import parse_cfg

    hydra.utils.get_original_cwd = os.getcwd  # only used for work_dir (logs)
    base = OmegaConf.load(os.path.join(tdmpc2_dir, "config.yaml"))
    base.pop("defaults", None)  # hydra launcher override; irrelevant outside hydra
    over = dict(task=task, checkpoint=checkpoint, model_size=5,
                eval_episodes=int(pc["episodes"]), seed=int(pc["eval_seed"]),
                compile=bool(pc["compile"]), save_video=False, enable_wandb=False)
    return parse_cfg(OmegaConf.merge(base, OmegaConf.create(over)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    require_prereg()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pc = cfg["planner_check"]
    src = os.path.join(ROOT, cfg["paths"]["tdmpc2_src"])
    rc, head = git("rev-parse", "HEAD", cwd=src)
    if rc != 0 or head != cfg["source"]["repo_commit"]:
        sys.exit(f"tdmpc2 at {src} is {head!r}, expected {cfg['source']['repo_commit']}.")
    tdmpc2_dir = os.path.join(src, "tdmpc2")
    sys.path.insert(0, tdmpc2_dir)
    os.chdir(tdmpc2_dir)

    import torch
    if not torch.cuda.is_available():
        sys.exit("a CUDA GPU is required (tdmpc2 hard-codes cuda:0).")
    from common.seed import set_seed
    from envs import make_env
    from tdmpc2 import TDMPC2

    torch.backends.cudnn.benchmark = True  # as evaluate.py
    rows, record = [], {}
    for item in pc["checkpoints"]:
        task, seed = item["task"], int(item["seed"])
        path = find_checkpoint(cfg, task, seed)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        sd = ck["model"] if "model" in ck else ck
        layout = layouts.detect_layout({k: v.detach().cpu().numpy() for k, v in sd.items()
                                        if hasattr(v, "detach")}, f"{task} seed {seed}")
        if layout != layouts.PUBLIC or cfg["survey"][task][seed] != layouts.PUBLIC:
            sys.exit(f"{task} seed {seed}: layout {layout}; the public code loads only the public layout.")

        c = tdmpc2_cfg(tdmpc2_dir, task, path, pc)
        set_seed(c.seed)
        env = make_env(c)
        agent = TDMPC2(c)
        agent.load(ck)  # dict path: state_dict["model"], api_model_conversion, load_state_dict

        pub, pub_step = published_return(src, task, seed)
        for eval_mode in pc["eval_modes"]:
            set_seed(c.seed)
            ep_returns, ep_times, ep_steps = [], [], []
            for i in range(c.eval_episodes):
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                obs, done, ep_reward, t = env.reset(), False, 0.0, 0
                while not done:
                    action = agent.act(obs, t0=t == 0, eval_mode=bool(eval_mode))
                    obs, reward, done, info = env.step(action)
                    ep_reward += float(reward)
                    t += 1
                torch.cuda.synchronize()
                ep_times.append(time.perf_counter() - t_start)
                ep_returns.append(ep_reward)
                ep_steps.append(t)
                print(f"{task} seed {seed} eval_mode={eval_mode} episode {i}: return "
                      f"{ep_reward:.1f}, {t} steps, {ep_times[-1]:.1f} s")
            R = np.array(ep_returns)
            rows.append(dict(task=task, checkpoint_seed=seed, eval_seed=c.seed, layout=layout,
                             eval_mode=bool(eval_mode), episodes=len(R), return_mean=R.mean(),
                             return_min=R.min(), return_max=R.max(), published=pub,
                             published_step=pub_step, fraction=R.mean() / pub,
                             seconds_first_episode=ep_times[0],
                             seconds_per_episode_after_first=(float(np.mean(ep_times[1:]))
                                                              if len(ep_times) > 1 else np.nan),
                             steps_per_episode=int(np.mean(ep_steps)), compile=c.compile,
                             iterations=c.iterations, num_samples=c.num_samples,
                             horizon=c.horizon))
            record[f"{task}/{seed}/eval_mode={eval_mode}"] = dict(
                checkpoint=path, returns=ep_returns, seconds=ep_times, steps=ep_steps)
            print(f"{task} seed {seed} eval_mode={eval_mode}: mean return {R.mean():.1f} vs "
                  f"published {pub} (fraction {R.mean() / pub:.3f})")

    out = os.path.join(ROOT, cfg["paths"]["results_out"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "planner_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    versions = {}
    for mod in ("torch", "tensordict", "torchrl", "hydra", "omegaconf", "mujoco", "dm_control",
                "gymnasium", "numpy"):
        try:
            versions[mod] = getattr(__import__(mod), "__version__", "unknown")
        except ImportError:
            versions[mod] = None
    meta = dict(stage="planner_check", git_commit=git("rev-parse", "HEAD")[1],
                git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")[1]),
                prereg_tags={t: git("rev-parse", t)[1] for t, _ in TAGS},
                tdmpc2_commit=head, gpu=torch.cuda.get_device_name(0),
                cuda=torch.version.cuda, python=platform.python_version(), versions=versions,
                time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                config=cfg, episodes=record)
    with open(os.path.join(out, "meta_planner_check.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {os.path.relpath(os.path.join(out, 'planner_check.csv'), ROOT)}")


if __name__ == "__main__":
    main()
