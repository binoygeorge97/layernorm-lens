"""R6, steps 2-4: list and fetch TD-MPC2 checkpoints, extract the first encoder
layer, compute its lens. Evaluation only: no observations, no criterion.

    python experiments/r6_tdmpc2/extract.py --config experiments/r6_tdmpc2/config.yaml list
    python experiments/r6_tdmpc2/extract.py --config experiments/r6_tdmpc2/config.yaml extract
    python experiments/r6_tdmpc2/extract.py --config experiments/r6_tdmpc2/config.yaml lens

Refuses to run unless the annotated tag prereg-r6 exists and prereg/r6.md in the
working tree is identical to the tagged version.

list     print every file in the Hugging Face repo that matches a configured task,
         the seeds found and the repo revision. Downloads nothing.
extract  download the configured seeds (pinned to source.hf_revision), load each
         with torch.load(weights_only=True) on CPU, print the encoder's state_dict
         keys, save E, b, LayerNorm gamma/beta and eps (float64) to weights_out.
lens     lens/geometry.py on each saved layer: z*, principal widths, kappa,
         degenerate; per-task .npz in lens_out and lens_summary.csv.

Every stage writes meta_<stage>.json (config, git commit, versions) next to its
outputs.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lens import geometry as geo  # noqa: E402

TAG = "prereg-r6"
HF_API = "https://huggingface.co/api/models/{repo}"
HF_FILE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"


def git(*args, cwd=ROOT):
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout.strip()


def require_prereg():
    rc, kind = git("cat-file", "-t", f"refs/tags/{TAG}")
    if rc != 0:
        sys.exit(f"refusing to run: git tag {TAG} does not exist (CLAUDE.md).")
    if kind != "tag":
        sys.exit(f"refusing to run: {TAG} is a lightweight tag; an annotated tag is required.")
    rc, _ = git("diff", "--quiet", TAG, "--", "prereg/r6.md")
    if rc != 0:
        sys.exit(f"refusing to run: prereg/r6.md differs from the version tagged {TAG}.")


def hf_repo(cfg):
    return cfg["source"]["checkpoints"].rstrip("/").split("huggingface.co/")[1]


def http_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def match(files, task):
    """Checkpoint files for a task: name contains the task, ends in .pt; seed = trailing int."""
    out = []
    for f in files:
        base = os.path.basename(f)
        if task in f and base.endswith(".pt"):
            m = re.search(r"-(\d+)\.pt$", base)
            out.append(dict(path=f, seed=int(m.group(1)) if m else None))
    return out


def meta(cfg, stage, extra):
    rc, commit = git("rev-parse", "HEAD")
    _, dirty = git("status", "--porcelain", "--untracked-files=no")
    m = dict(stage=stage, git_commit=commit, git_dirty=bool(dirty),
             prereg_tag_object=git("rev-parse", TAG)[1],
             jax=jax.__version__, numpy=np.__version__, python=platform.python_version(),
             x64=bool(jax.config.read("jax_enable_x64")),
             time=datetime.datetime.now(datetime.timezone.utc).isoformat(), config=cfg)
    src = os.path.join(ROOT, "checkpoints", "tdmpc2_src")
    if os.path.isdir(src):
        m["tdmpc2_clone_commit"] = git("rev-parse", "HEAD", cwd=src)[1]
    m.update(extra)
    out = os.path.join(ROOT, cfg["paths"]["results_out"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"meta_{stage}.json"), "w") as f:
        json.dump(m, f, indent=2)


def stage_list(cfg):
    info = http_json(HF_API.format(repo=hf_repo(cfg)))
    files = sorted(s["rfilename"] for s in info.get("siblings", []))
    print(f"repo {hf_repo(cfg)}  revision {info.get('sha')}  ({len(files)} files)")
    found = {}
    for task in cfg["tasks"]:
        hits = match(files, task)
        found[task] = hits
        seeds = sorted({h["seed"] for h in hits if h["seed"] is not None})
        print(f"\n{task}: {len(hits)} file(s), seeds {seeds}")
        for h in hits:
            print(f"  {h['path']}")
    meta(cfg, "list", dict(hf_revision=info.get("sha"), files=files, matches=found))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_extract(cfg):
    import torch

    rev = cfg["source"].get("hf_revision")
    if not rev:
        sys.exit("set source.hf_revision in config.yaml from the `list` stage first.")
    eps = float(cfg["layer"]["layernorm_eps"])
    torch_eps = torch.nn.LayerNorm(8).eps
    if torch_eps != eps:
        sys.exit(f"installed torch's LayerNorm default eps {torch_eps} != config {eps}.")
    info = http_json(HF_API.format(repo=hf_repo(cfg)) + f"/revision/{rev}")
    files = [s["rfilename"] for s in info.get("siblings", [])]
    ck_dir = os.path.join(ROOT, cfg["paths"]["checkpoints"])
    w_dir = os.path.join(ROOT, cfg["paths"]["weights_out"])
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)
    pre = cfg["layer"]["key_prefix"]
    record = {}
    for task in cfg["tasks"]:
        for seed in cfg["seeds"]:
            hits = [{"path": f, "seed": seed} for f in files if os.path.basename(f) == f"{task}-{seed}.pt"]
            if len(hits) != 1:
                sys.exit(f"{task} seed {seed}: expected exactly one file, found {hits}.")
            path = hits[0]["path"]
            local = os.path.join(ck_dir, path)
            if not os.path.exists(local):
                os.makedirs(os.path.dirname(local), exist_ok=True)
                url = HF_FILE.format(repo=hf_repo(cfg), rev=rev, path=path)
                urllib.request.urlretrieve(url, local)
            digest = sha256(local)
            sd = torch.load(local, map_location="cpu", weights_only=True)
            sd = sd["model"] if "model" in sd else sd
            enc_keys = [k for k in sd if k.startswith("_encoder.")]
            print(f"\n{task} seed {seed}  ({path}, sha256 {digest[:12]})")
            for k in enc_keys:
                print(f"  {k:40s} {tuple(sd[k].shape)} {sd[k].dtype}")
            need = [f"{pre}.weight", f"{pre}.bias", f"{pre}.ln.weight", f"{pre}.ln.bias"]
            missing = [k for k in need if k not in sd]
            if missing:
                sys.exit(f"{task}: missing keys {missing}")
            t = {k: sd[k].detach().cpu().numpy().astype(np.float64) for k in need}
            E = t[f"{pre}.weight"]  # nn.Linear weight: (out, in) = (H, k)
            H, k = E.shape
            out = os.path.join(w_dir, f"{task}-seed{seed}.npz")
            np.savez(out, E=E, b=t[f"{pre}.bias"], ln_gamma=t[f"{pre}.ln.weight"],
                     ln_beta=t[f"{pre}.ln.bias"], eps=eps,
                     source_dtype=str(sd[f"{pre}.weight"].dtype))
            print(f"  -> H = {H}, k = {k}, saved {os.path.relpath(out, ROOT)}")
            record[f"{task}/{seed}"] = dict(file=path, sha256=digest, H=H, k=k,
                                            encoder_keys=enc_keys)
    meta(cfg, "extract", dict(torch=torch.__version__, hf_revision=rev, checkpoints=record))


def stage_lens(cfg):
    w_dir = os.path.join(ROOT, cfg["paths"]["weights_out"])
    l_dir = os.path.join(ROOT, cfg["paths"]["lens_out"])
    os.makedirs(l_dir, exist_ok=True)
    rows = []
    for task in cfg["tasks"]:
        for seed in cfg["seeds"]:
            w = np.load(os.path.join(w_dir, f"{task}-seed{seed}.npz"))
            L = geo.lens(w["E"], w["b"], float(w["eps"]))
            np.savez(os.path.join(l_dir, f"{task}-seed{seed}.npz"),
                     z_star=L.z_star, principal_widths=L.principal_widths,
                     principal_widths_eff=L.principal_widths_eff,
                     principal_dirs=L.principal_dirs, sing=L.sing, c_perp=L.c_perp,
                     norm_c_perp=L.norm_c_perp, kappa=L.kappa, phi=L.phi,
                     degenerate=L.degenerate, Sigma=L.Sigma, H=L.H, k=L.k, eps=L.eps)
            pw = L.principal_widths_eff if L.degenerate else L.principal_widths
            rows.append(dict(task=task, seed=seed, H=L.H, k=L.k, eps=L.eps,
                             degenerate=L.degenerate, norm_c_perp=L.norm_c_perp,
                             kappa=L.kappa, norm_z_star=float(np.linalg.norm(L.z_star)),
                             width_kind="eps_limited" if L.degenerate else "r_star",
                             width_min=float(pw.min()), width_median=float(np.median(pw)),
                             width_max=float(pw.max())))
    path = os.path.join(ROOT, cfg["paths"]["results_out"], "lens_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(r)
    meta(cfg, "lens", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("stage", choices=["list", "extract", "lens"])
    args = ap.parse_args()
    require_prereg()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dict(list=stage_list, extract=stage_extract, lens=stage_lens)[args.stage](cfg)


if __name__ == "__main__":
    main()
