"""Write the lens/core.py golden files. Run once, from the repo root:

    python -m tests.golden.make_golden            # refuses to overwrite
    python -m tests.golden.make_golden --force    # only after a reviewed change

Outputs tests/golden/<case>.npz and tests/golden/manifest.json (versions, commit).
"""

import argparse
import json
import os
import subprocess

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from tests.golden import cases

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    manifest = os.path.join(HERE, "manifest.json")
    if os.path.exists(manifest) and not args.force:
        raise SystemExit(f"{manifest} exists; refusing to overwrite (use --force).")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=HERE).stdout.strip()
    core_sha = subprocess.run(["git", "hash-object", "lens/core.py"], capture_output=True,
                              text=True, cwd=os.path.dirname(os.path.dirname(HERE))).stdout.strip()
    info = dict(versions=cases.versions(), commit=commit, core_py_blob=core_sha, cases={})
    for name in cases.CASES:
        out = cases.compute(name)
        np.savez(os.path.join(HERE, f"{name}.npz"), **out)
        info["cases"][name] = len(out)
        print(f"{name}: {len(out)} arrays")
    with open(manifest, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info["versions"]))


if __name__ == "__main__":
    main()
