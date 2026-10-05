"""Throughput of P-I training for 1, 2 and 3 parallel processes (prereg/p1.md, execution
configuration), on RANDOM targets (no quadrotor data), with launch.py's pinned
environment.

    python experiments/p1_quadrotor/throughput.py --n-procs 1 2 3 --steps 1000

Each process runs every grid configuration (arch × init × depth) for `steps` minibatch
steps after a compile-and-warm-up call, as the benchmark stage does; P copies run
concurrently. Per P: wall time, mean ms/step per configuration under that load, and the
aggregate throughput in grid-runs per hour (the grid weights every configuration
equally). Writes results/p1/throughput/throughput.csv and meta_throughput.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "experiments", "r6_tdmpc2")]


def worker(cfg_path, steps, out):
    import numpy as np
    import yaml
    import run as prun
    from lens import models
    from lens import train as tr
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    b, t = cfg["benchmark"], cfg["training"]
    rng = np.random.default_rng(int(b["seed"]))
    W1, W2 = rng.standard_normal((16, 64)) / 4.0, rng.standard_normal((64, 12)) / 8.0
    Z, Zv = rng.standard_normal((int(b["n_train"]), 16)), rng.standard_normal((int(b["n_val"]), 16))
    data = dict(Z=Z, Y=np.tanh(Z @ W1) @ W2, Zv=Zv, Yv=np.tanh(Zv @ W1) @ W2)
    rows, seen = [], set()
    t_all = time.time()
    for name, spec, seed in prun.grid(cfg):
        key = (spec.arch, spec.init, spec.n_blocks)
        if key in seen:
            continue
        seen.add(key)
        p0 = models.init_params(spec, seed)
        base = dict(lr=float(t["lr"]), eval_every=int(t["eval_every"]), patience_frac=None, tol=float(t["tol"]),
                    batch_size=t["batch_size"], seed=seed, log_every=0)
        tr.train(spec, p0, data, dict(base, max_steps=int(t["eval_every"])))
        _, _, info = tr.train(spec, p0, data, dict(base, max_steps=steps))
        rows.append(dict(arch=spec.arch, init=spec.init, n_blocks=spec.n_blocks, ms_per_step=1e3 * info["seconds"] / steps))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, seconds=time.time() - t_all), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--n-procs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args.config, args.steps, args.worker)
        return
    import launch
    import numpy as np
    import yaml
    import run as prun
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = os.path.join(ROOT, *cfg["paths"]["results"].split("/"), "throughput")
    os.makedirs(out_dir, exist_ok=True)
    table, per_cfg = [], {}
    for P in args.n_procs:
        outs = [os.path.join(out_dir, f"worker_P{P}_{i}.json") for i in range(P)]
        cmds = [[sys.executable, __file__, "--config", args.config, "--steps", str(args.steps), "--worker", o]
                for o in outs]
        t0 = time.time()
        procs = [subprocess.Popen(c, env=launch.env(), cwd=ROOT) for c in cmds]
        codes = [p.wait() for p in procs]
        wall = time.time() - t0
        if any(codes):
            sys.exit(f"throughput: a worker failed for P = {P}: {codes}")
        res = [json.load(open(o, encoding="utf-8")) for o in outs]
        ms = np.mean([[r["ms_per_step"] for r in w["rows"]] for w in res], axis=0)
        per_cfg[P] = [dict(arch=r["arch"], init=r["init"], n_blocks=r["n_blocks"], ms_per_step=float(m))
                      for r, m in zip(res[0]["rows"], ms)]
        mean_ms = float(np.mean(ms))
        grid_hours = len(prun.grid(cfg)) * int(cfg["training"]["max_steps"]) * mean_ms / 1e3 / 3600 / P
        table.append(dict(n_proc=P, wall_seconds=wall, mean_ms_per_step=mean_ms,
                          steps_per_second_total=P * 1e3 / mean_ms, grid_hours_upper_bound=grid_hours))
        print(table[-1], flush=True)
        for o in outs:
            os.remove(o)
    prun.write_csv(os.path.join(out_dir, "throughput.csv"), table)
    prun.write_json(os.path.join(out_dir, "meta_throughput.json"),
                    prun.meta(cfg, "throughput", dict(steps=args.steps, env=launch.ENV, per_config=per_cfg,
                                                      table=table)))


if __name__ == "__main__":
    main()
