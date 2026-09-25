"""Regenerate the lost D4 policy-prior observations by rerunning the recorded CPU run.

    python experiments/r6_tdmpc2/regenerate_d4.py --config experiments/r6_tdmpc2/config.yaml run
    python experiments/r6_tdmpc2/regenerate_d4.py --config experiments/r6_tdmpc2/config.yaml compare
    python experiments/r6_tdmpc2/regenerate_d4.py --config experiments/r6_tdmpc2/config.yaml publish --drive DIR

The observations of the D4 collection (data/r6/<task>-seed<seed>.npz) were never
copied off the Colab CPU runtime and are lost; meta_collect.json records no data
hashes. The D6 encoder agreement gate needs them. This script reruns that run's own
code, unchanged, and checks that it reproduces the committed results exactly.

Refuses to run unless the annotated tags prereg-r6, -d1, -d2 and -d3 exist and the
pre-registration files match them. Computes no criterion quantity (Inside,
Populated, Sharp), no gate G1 and nothing of D6 (b), (c) or (g): the only numbers
it produces are those the recorded run produced (returns, D3 consistency, lens).

run      Check out d4_regeneration.run_commit (the commit recorded in all four meta
         files of the CPU run) in a separate git worktree. Download the 15
         checkpoints from the pinned Hugging Face revision and refuse any file whose
         SHA-256 differs from D6's table (read from the tag prereg-r6-d3). Record the
         installed package versions next to those recorded in meta_collect.json.
         Then run, in the worktree and with that commit's config, exactly the stages
         of the original run: extract.py list, extract, lens; collect.py collect,
         consistency. Logs, exit codes and timings go to results/r6/d4_regen/.
compare  Compare the regenerated returns.csv and consistency.csv with the committed
         ones, value by value at their full printed precision (the pass rule), with
         byte-level differences reported separately. For information only, also
         compare consistency.json (full float64 precision) and lens_summary.csv, and
         the checkpoint SHA-256s in meta_extract.json. Prints MATCH (exit 0) or
         DIFFERS (exit 3); on DIFFERS, stop and consult the author: substitute data
         needs a D7 deviation before the gate may use it.
publish  Copy the 15 observation files to <drive>/data/r6/ (verifying each copy's
         SHA-256; an existing file with another hash is never overwritten), write
         results/r6/d4_obs_manifest.csv and results/r6/meta_d4_regeneration.json,
         copy the regenerated summaries to results/r6/d4_regen/ and all of it to
         <drive>/results/r6/, and print the exact `git add -f` command.
"""

import argparse
import datetime
import glob
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

import jax

jax.config.update("jax_enable_x64", True)  # CLAUDE.md; this script computes nothing itself
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import provenance as pv  # noqa: E402

ROOT = pv.ROOT
REGEN = os.path.join(ROOT, "results", "r6", "d4_regen")
HF_FILE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"
PACKAGES = ["jax", "jaxlib", "numpy", "scipy", "torch", "mujoco", "dm_control", "PyYAML",
            "matplotlib", "pytest"]


def die(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def worktree(rc):
    return os.path.abspath(rc["worktree"] if os.path.isabs(rc["worktree"])
                           else os.path.join(ROOT, rc["worktree"]))


def versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def all_distributions():
    return sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())


def recorded_versions():
    """Versions recorded by the original collect stage (meta_collect.json at HEAD)."""
    rc, text = pv.git("show", "HEAD:results/r6/meta_collect.json")
    if rc != 0:
        die("results/r6/meta_collect.json is not committed at HEAD.")
    m = json.loads(text)
    return {k: m.get(k) for k in ("python", "torch", "jax", "numpy", "mujoco", "dm_control",
                                  "git_commit", "time")}


# --------------------------------------------------------------------------- #
# run                                                                           #
# --------------------------------------------------------------------------- #


def setup_worktree(cfg, rc):
    wt = worktree(rc)
    commit = rc["run_commit"]
    if not os.path.isdir(wt):
        r = subprocess.run(["git", "worktree", "add", "--detach", wt, commit], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            die(f"git worktree add failed: {r.stderr.strip()}")
    head = pv.git("rev-parse", "HEAD", cwd=wt)[1]
    if head != commit:
        die(f"worktree {wt} is at {head}, expected {commit}.")
    if pv.git("status", "--porcelain", "--untracked-files=no", cwd=wt)[1]:
        die(f"worktree {wt} has modified tracked files; remove it and rerun.")
    # the recorded run's config and code, as committed
    wcfg = yaml.safe_load(pv.git("show", f"{commit}:experiments/r6_tdmpc2/config.yaml")[1])
    # tdmpc2 source (published returns), shared with the main checkout
    src = os.path.join(ROOT, cfg["paths"]["tdmpc2_src"])
    want = cfg["source"]["repo_commit"]
    if pv.git("rev-parse", "HEAD", cwd=src)[1] != want:
        die(f"tdmpc2 at {src} is not at {want}; clone it first (notebook cell 5).")
    link = os.path.join(wt, wcfg["paths"]["tdmpc2_src"].rstrip("/"))
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if not os.path.lexists(link):
        os.symlink(src, link)
    if pv.git("rev-parse", "HEAD", cwd=link)[1] != want:
        die(f"{link} does not resolve to tdmpc2 at {want}.")
    return wt, wcfg


def download_checkpoints(wt, wcfg, table):
    """The 15 files from the pinned revision; refuse any SHA-256 not in D6's table."""
    repo = wcfg["source"]["checkpoints"].rstrip("/").split("huggingface.co/")[1]
    rev = wcfg["source"]["hf_revision"]
    ck_dir = os.path.join(wt, wcfg["paths"]["checkpoints"])
    out = {}
    for task in wcfg["tasks"]:
        for seed in wcfg["seeds"]:
            name = pv.checkpoint_name(task, seed)
            if name not in table:
                die(f"{name} is not in D6's table.")
            local = os.path.join(ck_dir, name)
            if not os.path.exists(local):
                os.makedirs(os.path.dirname(local), exist_ok=True)
                url = HF_FILE.format(repo=repo, rev=rev, path=name)
                part = local + ".part"
                urllib.request.urlretrieve(url, part)
                try:
                    pv.verify_sha256(part, table[name], name)
                except pv.ProvenanceError as e:
                    os.remove(part)
                    die(str(e))
                os.replace(part, local)
            try:
                out[name] = pv.verify_sha256(local, table[name], name)
            except pv.ProvenanceError as e:
                die(f"{e} Delete {local} and rerun.")
            print(f"  {name}  sha256 {out[name]}  ok")
    return dict(repo=repo, revision=rev, sha256=out)


def run_stage(wt, script, stage, log_dir):
    cmd = [sys.executable, f"experiments/r6_tdmpc2/{script}.py", "--config",
           "experiments/r6_tdmpc2/config.yaml", stage]
    env = dict(os.environ, MUJOCO_GL="egl", PYTHONUNBUFFERED="1")
    log = os.path.join(log_dir, f"{script}_{stage}.log")
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=wt, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        p.wait()
    return dict(command=" ".join(cmd[1:]), exit=p.returncode, seconds=time.time() - t0,
                log=os.path.relpath(log, ROOT))


def stage_run(cfg):
    rc = cfg["d4_regeneration"]
    table = pv.d6_sha_table()
    wt, wcfg = setup_worktree(cfg, rc)
    os.makedirs(os.path.join(REGEN, "logs"), exist_ok=True)
    rec = recorded_versions()
    now = dict(python=platform.python_version(), **versions())
    print("package versions, installed vs recorded (meta_collect.json):")
    for k in ("python", "torch", "jax", "numpy", "mujoco", "dm_control"):
        print(f"  {k:10s} {str(now.get(k)):22s} recorded {rec.get(k)}")
    print(f"checkpoints -> {os.path.join(wt, wcfg['paths']['checkpoints'])}")
    ck = download_checkpoints(wt, wcfg, table)
    stages = []
    for script, stage in [tuple(s.split()) for s in rc["stages"]]:
        print(f"\n=== {script}.py {stage} (worktree {pv.git('rev-parse', '--short', 'HEAD', cwd=wt)[1]}) ===")
        r = run_stage(wt, script, stage, os.path.join(REGEN, "logs"))
        expected = int(rc["expected_exit"][f"{script} {stage}"])
        r.update(expected_exit=expected, as_expected=r["exit"] == expected)
        stages.append(r)
        print(f"--- exit {r['exit']} (expected {expected}), {r['seconds']:.0f} s")
        if not r["as_expected"]:
            _write_json(os.path.join(REGEN, "run.json"),
                        _run_record(cfg, wt, ck, now, rec, stages, complete=False))
            die(f"{script}.py {stage} exited {r['exit']}, expected {expected}: stopping. "
                f"See {r['log']}.")
    _write_json(os.path.join(REGEN, "run.json"),
                _run_record(cfg, wt, ck, now, rec, stages, complete=True))
    print(f"\nrun complete; next: compare.")


def _run_record(cfg, wt, ck, now, rec, stages, complete):
    return dict(stage="d4_regeneration_run", complete=complete,
                run_commit=cfg["d4_regeneration"]["run_commit"],
                worktree=wt, worktree_state=pv.git_state(cwd=wt),
                python=platform.python_version(), executable=sys.executable,
                platform=platform.platform(), versions=now, recorded_versions=rec,
                distributions=all_distributions(), checkpoints=ck, stages=stages,
                time=datetime.datetime.now(datetime.timezone.utc).isoformat())


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# compare                                                                       #
# --------------------------------------------------------------------------- #


def _committed(path):
    """Bytes of a committed file at HEAD, and its blob id."""
    r = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, cwd=ROOT)
    if r.returncode != 0:
        die(f"{path} is not committed at HEAD.")
    return r.stdout, pv.git("rev-parse", f"HEAD:{path}")[1]


def _json_diffs(a, b, path=""):
    """Paths where two JSON values differ (floats compared exactly)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: missing in {'new' if k in a else 'ref'}" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += _json_diffs(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _json_diffs(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def stage_compare(cfg):
    rc = cfg["d4_regeneration"]
    run = json.load(open(os.path.join(REGEN, "run.json")))
    if not run.get("complete"):
        die("the run stage did not complete; rerun it first.")
    new_dir = os.path.join(run["worktree"], "results", "r6")
    result = dict(stage="d4_regeneration_compare", pass_rule=rc["compare"]["pass"],
                  files={}, info={})
    for name in rc["compare"]["pass"]:
        ref, blob = _committed(f"results/r6/{name}")
        c = pv.compare_csv(ref, open(os.path.join(new_dir, name), "rb").read())
        result["files"][name] = dict(reference_blob=blob, **c)
    for name in rc["compare"]["info_csv"]:
        ref, blob = _committed(f"results/r6/{name}")
        c = pv.compare_csv(ref, open(os.path.join(new_dir, name), "rb").read())
        result["info"][name] = dict(reference_blob=blob, **c)
    for name in rc["compare"]["info_json"]:
        ref, blob = _committed(f"results/r6/{name}")
        d = _json_diffs(json.loads(ref), json.load(open(os.path.join(new_dir, name))))
        result["info"][name] = dict(reference_blob=blob, equal=not d, diffs=d[:200],
                                    n_diffs=len(d))
    ref_ext = json.loads(_committed("results/r6/meta_extract.json")[0])["checkpoints"]
    new_ext = json.load(open(os.path.join(new_dir, "meta_extract.json")))["checkpoints"]
    result["info"]["meta_extract.json sha256"] = dict(
        equal=all(ref_ext[k]["sha256"] == new_ext.get(k, {}).get("sha256") for k in ref_ext),
        n=len(ref_ext))
    result["match"] = all(f["values_equal"] for f in result["files"].values())
    result["time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_json(os.path.join(REGEN, "compare.json"), result)

    print("pass rule: every value of returns.csv and consistency.csv equal at printed precision")
    for name, f in result["files"].items():
        print(f"  {name}: values {'EQUAL' if f['values_equal'] else 'DIFFER'} "
              f"({f['n_rows_new']} rows vs {f['n_rows_ref']}), bytes "
              f"{'identical' if f['bytes_equal'] else 'differ: ' + f['byte_note']}")
        if f["header_ref"] != f["header_new"]:
            print(f"    header: ref {f['header_ref']}\n            new {f['header_new']}")
        for row, col, a, b in f["cell_diffs"][:50]:
            print(f"    row {row} {col}: committed {a!r}, regenerated {b!r}")
    print("information only (not part of the pass rule):")
    for name, f in result["info"].items():
        eq = f.get("values_equal", f.get("equal"))
        print(f"  {name}: {'equal' if eq else 'DIFFER'}"
              + (f", bytes {'identical' if f['bytes_equal'] else 'differ'}" if "bytes_equal" in f else "")
              + (f", {f['n_diffs']} differences" if f.get("n_diffs") else ""))
    if result["match"]:
        print("\nMATCH: the regenerated run reproduces the committed results exactly.")
    else:
        print("\nDIFFERS: stop and consult the author. Substitute data needs a D7 deviation "
              "before the gate may use it.")
        sys.exit(3)


# --------------------------------------------------------------------------- #
# publish                                                                       #
# --------------------------------------------------------------------------- #


def _copy_verified(src, dst):
    """Copy src to dst and verify the copy; never overwrite a different file."""
    h = pv.sha256(src)
    if os.path.exists(dst):
        if pv.sha256(dst) != h:
            die(f"{dst} exists with a different SHA-256; not overwriting. Move it aside first.")
        return h, "already present"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".part")
    if pv.sha256(dst + ".part") != h:
        os.remove(dst + ".part")
        die(f"copy of {src} to {dst} does not verify.")
    os.replace(dst + ".part", dst)
    return h, "copied"


def stage_publish(cfg, drive):
    rc = cfg["d4_regeneration"]
    if not drive or not os.path.isdir(drive):
        die(f"--drive {drive!r} is not a directory (mount Google Drive first).")
    run = json.load(open(os.path.join(REGEN, "run.json")))
    cmp_path = os.path.join(REGEN, "compare.json")
    if not run.get("complete") or not os.path.exists(cmp_path):
        die("run and compare must both have completed.")
    cmp = json.load(open(cmp_path))
    wt = run["worktree"]
    wcfg = yaml.safe_load(pv.git("show", f"{rc['run_commit']}:experiments/r6_tdmpc2/config.yaml")[1])
    data_dir = os.path.join(wt, wcfg["paths"]["data_out"])
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    want = sorted(f"{t}-seed{s}.npz" for t in wcfg["tasks"] for s in wcfg["seeds"])
    if [os.path.basename(f) for f in files] != want:
        die(f"{data_dir}: expected {want}, found {[os.path.basename(f) for f in files]}.")

    manifest = os.path.join(ROOT, "results", "r6", "d4_obs_manifest.csv")
    rows = pv.write_manifest(manifest, files, data_dir)
    drive_data = os.path.join(drive, "data", "r6")
    copies = {}
    for f in files:
        copies[os.path.basename(f)] = _copy_verified(f, os.path.join(drive_data, os.path.basename(f)))[1]
        print(f"  data/r6/{os.path.basename(f)} -> {drive_data}: {copies[os.path.basename(f)]}")

    # the regenerated run's own summaries and meta files, for the record
    out = []
    for name in rc["summaries"]:
        src = os.path.join(wt, "results", "r6", name)
        dst = os.path.join(REGEN, name)
        shutil.copyfile(src, dst)
        out.append(dst)
    meta_path = os.path.join(ROOT, "results", "r6", "meta_d4_regeneration.json")
    meta = dict(stage="d4_regeneration", **pv.git_state(), prereg_tags=pv.tag_objects(),
                run_commit=rc["run_commit"], jax=jax.__version__,
                torch=run["versions"].get("torch"), python=platform.python_version(),
                versions=run["versions"], recorded_versions=run["recorded_versions"],
                match=cmp["match"],
                compare={k: dict(values_equal=v["values_equal"], bytes_equal=v["bytes_equal"],
                                 byte_note=v["byte_note"], n_cell_diffs=len(v["cell_diffs"]))
                         for k, v in cmp["files"].items()},
                checkpoints=run["checkpoints"], stages=run["stages"],
                data=dict(files=rows, drive=drive_data, copies=copies,
                          manifest="results/r6/d4_obs_manifest.csv",
                          manifest_sha256=pv.sha256(manifest)),
                config=rc, time=datetime.datetime.now(datetime.timezone.utc).isoformat())
    _write_json(meta_path, meta)

    commit_files = ([manifest, meta_path, os.path.join(REGEN, "run.json"), cmp_path] + out
                    + sorted(glob.glob(os.path.join(REGEN, "logs", "*.log"))))
    drive_res = os.path.join(drive, "results", "r6")
    for p in commit_files:
        dst = os.path.join(drive_res, os.path.relpath(p, os.path.join(ROOT, "results", "r6")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(p, dst)
    big = [p for p in commit_files if os.path.getsize(p) >= 1 << 20]
    if big:
        print(f"WARNING: over 1 MB, do not commit as is: {big}")
    print(f"\n{'MATCH' if cmp['match'] else 'DIFFERS (do not use this data before a D7 deviation)'}")
    print(f"observations: {len(files)} files on {drive_data}; manifest {os.path.relpath(manifest, ROOT)}")
    print("\nfiles to commit:")
    print(pv.git_add_command(commit_files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--drive", help="publish: the Drive folder MyDrive/layernorm-lens-r6")
    ap.add_argument("stage", choices=["run", "compare", "publish"])
    args = ap.parse_args()
    try:
        pv.require_prereg()
    except pv.ProvenanceError as e:
        die(str(e))
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.stage == "run":
        stage_run(cfg)
    elif args.stage == "compare":
        stage_compare(cfg)
    else:
        stage_publish(cfg, args.drive)


if __name__ == "__main__":
    main()
