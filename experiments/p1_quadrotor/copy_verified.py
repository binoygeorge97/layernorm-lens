"""Copy a P-I output directory to another location (e.g. Google Drive) and verify it.

    python experiments/p1_quadrotor/copy_verified.py SRC DEST

Every run writes to local disk (atomic renames need a local file system; Drive sync can
lock files and break os.replace on Windows). After each stage, this copies every file
under SRC to the same relative path under DEST with plain copies (no rename at the
destination), recomputes each copy's SHA-256 and compares it with the source's, and
writes copy_manifest.csv (file, bytes, sha256) to both SRC and DEST. It exits non-zero,
listing every problem, if any copy differs. Files already present at DEST with the right
hash are not copied again, so it can be rerun after an interruption.
"""

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "experiments", "r6_tdmpc2")]
import provenance as pv  # noqa: E402

MANIFEST = "copy_manifest.csv"


def copy_verified(src, dest):
    """Returns (rows, problems): rows of the manifest; problems, empty if every copy verifies."""
    files = []
    for d, _, names in os.walk(src):
        files += [os.path.join(d, n) for n in names if n != MANIFEST and ".tmp" not in n]
    rows, problems = [], []
    for f in sorted(files):
        rel = os.path.relpath(f, src)
        h = pv.sha256(f)
        out = os.path.join(dest, rel)
        if not (os.path.exists(out) and pv.sha256(out) == h):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copyfile(f, out)
        if pv.sha256(out) != h:
            problems.append(f"{rel}: SHA-256 mismatch after copying")
        rows.append(dict(file=rel.replace(os.sep, "/"), bytes=os.path.getsize(f), sha256=h))
    if not problems:
        for d in (src, dest):
            pv.write_manifest(os.path.join(d, MANIFEST), [os.path.join(src, r["file"]) for r in rows], src)
    return rows, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dest")
    args = ap.parse_args()
    rows, problems = copy_verified(args.src, args.dest)
    if problems:
        sys.exit("copy_verified: " + "; ".join(problems))
    print(f"{len(rows)} files copied to {args.dest} and verified (SHA-256); {MANIFEST} written to both")


if __name__ == "__main__":
    main()
