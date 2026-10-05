"""Launcher for P-I's training runs on the laptop (prereg/p1.md, execution configuration).

    python experiments/p1_quadrotor/launch.py --stage train       [--n-proc 3] [-- extra run.py args]
    python experiments/p1_quadrotor/launch.py --stage train_long  [--n-proc 3]

Runs every member of the stage's grid as its own process (`run.py STAGE --index i`), at
most N_PROC at a time, each with the pinned environment `ENV` (one thread per process
for XLA and every BLAS). Each process skips a run whose manifest already verifies and
restarts any other run from step 0 (run.stage_train), so the launcher can simply be
started again after an interruption. Logs go to <results>/<stage>/logs/<name>.log. Exit
status is non-zero if any run's process failed.

The configuration was chosen by measuring throughput for 1, 2 and 3 processes
(throughput.py, results/p1/throughput/) and checking determinism (determinism.py,
results/p1/determinism/): see docs/DECISIONS.md.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

N_PROC = 3
THREADS = 1
ENV = {
    "XLA_FLAGS": f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={THREADS}",
    "OMP_NUM_THREADS": str(THREADS),
    "MKL_NUM_THREADS": str(THREADS),
    "OPENBLAS_NUM_THREADS": str(THREADS),
    "NUMEXPR_NUM_THREADS": str(THREADS),
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONUTF8": "1",
}


def env(base=None):
    """The process environment: `base` (default os.environ) with ENV pinned on top."""
    e = dict(os.environ if base is None else base)
    e.update(ENV)
    return e


def command(script, config, stage, index, extra=()):
    return [sys.executable, script, "--config", config, stage, "--index", str(index), *extra]


def run_pool(cmds, n_proc, log_paths, poll=2.0):
    """Run the commands, at most n_proc at a time, each with env(); returns the exit codes
    in order."""
    pending = list(range(len(cmds)))
    running, codes = {}, [None] * len(cmds)
    while pending or running:
        while pending and len(running) < n_proc:
            i = pending.pop(0)
            os.makedirs(os.path.dirname(log_paths[i]), exist_ok=True)
            f = open(log_paths[i], "w", encoding="utf-8")
            running[i] = (subprocess.Popen(cmds[i], env=env(), stdout=f, stderr=subprocess.STDOUT, cwd=ROOT), f)
        for i, (proc, f) in list(running.items()):
            if proc.poll() is not None:
                f.close()
                codes[i] = proc.returncode
                print(f"[{time.strftime('%H:%M:%S')}] {os.path.basename(log_paths[i])}: exit {proc.returncode}",
                      flush=True)
                del running[i]
        time.sleep(poll)
    return codes


def main():
    sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "experiments", "r6_tdmpc2")]
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "train_long"])
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--script", default=os.path.join(HERE, "run.py"))
    ap.add_argument("--n-proc", type=int, default=N_PROC)
    ap.add_argument("--logs", default=None, help="log directory (default <results>/<stage>/logs)")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after --: passed to every run")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    import run as prun  # noqa: E402
    members = prun.grid(cfg, prun.GRIDS[args.stage])
    extra = [a for a in args.extra if a != "--"]
    logs = args.logs or os.path.join(ROOT, *cfg["paths"]["results"].split("/"), args.stage, "logs")
    cmds = [command(args.script, args.config, args.stage, i, extra) for i in range(len(members))]
    codes = run_pool(cmds, args.n_proc, [os.path.join(logs, f"{n}.log") for n, _, _ in members])
    bad = [members[i][0] for i, c in enumerate(codes) if c != 0]
    print(f"{len(members) - len(bad)} of {len(members)} runs exited 0" + (f"; failed: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
