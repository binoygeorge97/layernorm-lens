"""Determinism of P-I training under the launcher's environment (prereg/p1.md, execution
configuration; author's item A7), on the pre-flight's SYNTHETIC linear plant.

    python experiments/p1_quadrotor/determinism.py

Two runs of the real configuration, cut to 2,000 steps (preflight_override.yaml), are
trained three times each, every time in a fresh process with launch.env():
  A: alone; B: alone again (consecutive); C: alongside two other training processes.
Each run's outputs that do not record wall time (best parameters, history CSV, history
NPZ) must have identical SHA-256 across A, B and C. (The run JSON records seconds and a
timestamp, and the manifest lists the JSON's hash, so those two differ by design.)
Writes results/p1/determinism/determinism.json.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "experiments", "r6_tdmpc2")]

MEMBERS = ("prenorm-zero_bias-b3-s0", "normedlinear-torch_default-b3-s1")
LOAD = ("normedlinear-zero_bias-b3-s2", "prenorm-torch_default-b3-s3")


def worker(name, out_root):
    import preflight
    import p1_data as pdata
    import p1_load
    prun = p1_load.run()
    cfg = preflight.load_cfg(os.path.join(HERE, "config.yaml"))
    P = prun.paths(cfg)
    ds = pdata.load_dataset(P["data"], P["manifest"])
    member = [m for m in prun.grid(cfg) if m[0] == name]
    prun.stage_train(cfg, ds, os.path.join(out_root, "ck"), os.path.join(out_root, "res"), member, base=out_root,
                     summary=False, skip_complete=False)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        return
    import launch
    import preflight
    import provenance as pv
    import p1_load
    prun = p1_load.run()
    cfg = preflight.load_cfg(os.path.join(HERE, "config.yaml"))
    P = prun.paths(cfg)
    if not os.path.exists(P["manifest"]):
        plant, hx, hu = preflight.synthetic_plant(cfg)
        prun.stage_generate(cfg, plant, hx, hu, P["data"], P["manifest"])
    scratch = os.path.join(P["res"], "determinism")

    def spawn(name, tag):
        return subprocess.Popen([sys.executable, __file__, "--worker", name, os.path.join(scratch, tag, name)],
                                env=launch.env(), cwd=ROOT)

    for tag in ("A", "B"):
        for name in MEMBERS:
            assert spawn(name, tag).wait() == 0
    for name in MEMBERS:
        procs = [spawn(name, "C")] + [spawn(other, "C_load") for other in LOAD]
        assert all(p.wait() == 0 for p in procs)
    out = {}
    for name in MEMBERS:
        files = {f"{name}.npz": ("ck", f"{name}.npz"), f"{name}.csv": ("res", "history", f"{name}.csv"),
                 f"{name}_history.npz": ("res", "history", f"{name}.npz")}
        h = {tag: {k: pv.sha256(os.path.join(scratch, tag, name, *v)) for k, v in files.items()} for tag in "ABC"}
        out[name] = dict(sha256=h, consecutive_identical=h["A"] == h["B"], alongside_identical=h["A"] == h["C"])
    ok = all(v["consecutive_identical"] and v["alongside_identical"] for v in out.values())
    res = os.path.join(ROOT, "results", "p1", "determinism")
    prun.write_json(os.path.join(res, "determinism.json"),
                    prun.meta(cfg, "determinism", dict(env=launch.ENV, members=list(MEMBERS), load=list(LOAD),
                                                       steps=cfg["training"]["max_steps"], results=out,
                                                       all_identical=ok)))
    print(json.dumps({k: (v["consecutive_identical"], v["alongside_identical"]) for k, v in out.items()}))
    print("all byte-identical:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
