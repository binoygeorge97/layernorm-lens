"""Provenance helpers shared by the R6 D4 regeneration and the D6 planner collector.

Standard library only (no numpy, JAX or torch), so every R6 environment can import
it: the project's JAX environment, the Python 3.13 CPU environment of the D4 run
and tdmpc2's pinned Python 3.11 environment.

- git(), require_prereg(): refuse to run unless annotated tags exist and the
  pre-registration files match them (CLAUDE.md).
- d6_sha_table(): the checkpoint SHA-256 table of D6, read from the tag itself
  (git show prereg-r6-d3:prereg/r6-deviations-3.md), never from the working tree.
- sha256(), verify_sha256(), download_verified(): file hashing; a mismatch
  refuses the file.
- write_manifest(): file names, sizes and SHA-256s of data files that stay on Drive.
- compare_csv(): value-by-value comparison of two CSVs at their printed precision,
  with byte-level differences reported separately.
- git_add_command(): the exact `git add -f` command for a stage's result files.
"""

import csv
import hashlib
import io
import os
import re
import shlex
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TAGS = [("prereg-r6", "prereg/r6.md"), ("prereg-r6-d1", "prereg/r6-deviations.md"),
        ("prereg-r6-d2", "prereg/r6-deviations-2.md"),
        ("prereg-r6-d3", "prereg/r6-deviations-3.md"),
        ("prereg-r6-d4", "prereg/r6-deviations-4.md")]
D6_TAG, D6_PATH = "prereg-r6-d3", "prereg/r6-deviations-3.md"
N_CHECKPOINTS = 15


class ProvenanceError(RuntimeError):
    pass


def _run_git(args, cwd):
    # git writes UTF-8; without an explicit encoding Windows decodes it as cp1252.
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          encoding="utf-8", cwd=cwd)


def git(*args, cwd=ROOT):
    r = _run_git(args, cwd)
    return r.returncode, r.stdout.strip()


def require_prereg(tags=TAGS, cwd=ROOT):
    """Refuse unless every tag is annotated and its file matches the tagged version."""
    for tag, path in tags:
        rc, kind = git("cat-file", "-t", f"refs/tags/{tag}", cwd=cwd)
        if rc != 0:
            raise ProvenanceError(f"refusing to run: git tag {tag} does not exist (CLAUDE.md).")
        if kind != "tag":
            raise ProvenanceError(f"refusing to run: {tag} is a lightweight tag; "
                                  "an annotated tag is required.")
        if git("diff", "--quiet", tag, "--", path, cwd=cwd)[0] != 0:
            raise ProvenanceError(f"refusing to run: {path} differs from the version tagged {tag}.")


def require_tags_on_origin(tags=TAGS, remote="origin", cwd=ROOT):
    """Refuse unless every tag is on `remote`, annotated there (a peeled ^{} ref
    exists), with the same tag object as the local tag. Returns {tag: tag object}."""
    rc, out = git("ls-remote", "--tags", remote, cwd=cwd)
    if rc != 0:
        raise ProvenanceError(f"refusing to run: git ls-remote --tags {remote} failed.")
    refs = {}
    for ln in out.splitlines():
        if "\t" in ln:
            sha, ref = ln.split("\t", 1)
            refs[ref] = sha
    found = {}
    for tag, _ in tags:
        local = git("rev-parse", f"refs/tags/{tag}", cwd=cwd)[1]
        there = refs.get(f"refs/tags/{tag}")
        if there is None:
            raise ProvenanceError(f"refusing to run: tag {tag} is not on {remote}.")
        if f"refs/tags/{tag}^{{}}" not in refs:
            raise ProvenanceError(f"refusing to run: {tag} on {remote} is not an annotated tag.")
        if there != local:
            raise ProvenanceError(f"refusing to run: {tag} on {remote} is {there}, locally {local}.")
        found[tag] = there
    return found


def tag_objects(tags=TAGS, cwd=ROOT):
    return {t: dict(tag_object=git("rev-parse", t, cwd=cwd)[1],
                    commit=git("rev-parse", f"{t}^{{commit}}", cwd=cwd)[1]) for t, _ in tags}


def git_state(cwd=ROOT):
    return dict(git_commit=git("rev-parse", "HEAD", cwd=cwd)[1],
                git_dirty=bool(git("status", "--porcelain", "--untracked-files=no", cwd=cwd)[1]))


# --------------------------------------------------------------------------- #
# D6 checkpoint table                                                           #
# --------------------------------------------------------------------------- #

_ROW = re.compile(r"^\|\s*([0-9a-f]{64})\s*\|\s*(dmcontrol/[A-Za-z0-9_-]+-\d+\.pt)(?:\s*\(\*\))?\s*\|\s*$")


def parse_sha_table(text):
    """{'dmcontrol/<task>-<seed>.pt': sha256} from D6's 'Checkpoint provenance' table."""
    table = {}
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if m:
            sha, name = m.groups()
            if name in table:
                raise ProvenanceError(f"D6 table lists {name} twice.")
            table[name] = sha
    return table


def d6_sha_table(cwd=ROOT):
    """The D6 SHA-256 table, read from the annotated tag prereg-r6-d3 itself."""
    rc, kind = git("cat-file", "-t", f"refs/tags/{D6_TAG}", cwd=cwd)
    if rc != 0 or kind != "tag":
        raise ProvenanceError(f"annotated tag {D6_TAG} not found.")
    r = _run_git(["show", f"{D6_TAG}:{D6_PATH}"], cwd)
    if r.returncode != 0:
        raise ProvenanceError(f"git show {D6_TAG}:{D6_PATH} failed: {r.stderr.strip()}")
    table = parse_sha_table(r.stdout)
    if len(table) != N_CHECKPOINTS:
        raise ProvenanceError(f"D6 table has {len(table)} rows, expected {N_CHECKPOINTS}.")
    return table


def checkpoint_name(task, seed):
    return f"dmcontrol/{task}-{seed}.pt"


# --------------------------------------------------------------------------- #
# hashing and manifests                                                         #
# --------------------------------------------------------------------------- #


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path, expected, what=None):
    """Return the file's SHA-256; raise if it is not `expected`."""
    got = sha256(path)
    if got != expected:
        raise ProvenanceError(f"refusing {what or path}: SHA-256 {got} != expected {expected}.")
    return got


HF_FILE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"


def download_verified(url, local, expected, what=None):
    """Download url to local unless present; refuse (and delete a fresh download)
    unless the file's SHA-256 is `expected`. Returns the SHA-256."""
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        part = local + ".part"
        urllib.request.urlretrieve(url, part)
        try:
            verify_sha256(part, expected, what)
        except ProvenanceError:
            os.remove(part)
            raise
        os.replace(part, local)
    return verify_sha256(local, expected, what)


def write_manifest(path, files, base):
    """CSV of file (relative to `base`), bytes and sha256 for each path in `files`."""
    rows = [dict(file=os.path.relpath(p, base), bytes=os.path.getsize(p), sha256=sha256(p))
            for p in sorted(files)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    return rows


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# CSV comparison                                                                #
# --------------------------------------------------------------------------- #


def compare_csv(ref_bytes, new_bytes):
    """Compare two CSVs cell by cell as printed (their full printed precision).

    Returns dict(values_equal, bytes_equal, header_ref, header_new, n_rows_ref,
    n_rows_new, cell_diffs=[(row, column, ref, new)], byte_note). values_equal
    requires the same header, the same number of rows and every cell string-equal.
    bytes_equal is reported separately (line endings, trailing newline).
    """
    def rows(b):
        return list(csv.reader(io.StringIO(b.decode("utf-8"), newline="")))

    r, n = rows(ref_bytes), rows(new_bytes)
    out = dict(bytes_equal=ref_bytes == new_bytes, header_ref=r[0] if r else [],
               header_new=n[0] if n else [], n_rows_ref=max(len(r) - 1, 0),
               n_rows_new=max(len(n) - 1, 0), cell_diffs=[], byte_note="")
    same_shape = out["header_ref"] == out["header_new"] and len(r) == len(n)
    if same_shape:
        cols = out["header_ref"]
        for i, (a, b) in enumerate(zip(r[1:], n[1:])):
            if len(a) != len(b):
                out["cell_diffs"].append((i, "<row length>", len(a), len(b)))
                continue
            for c, x, y in zip(cols, a, b):
                if x != y:
                    out["cell_diffs"].append((i, c, x, y))
    out["values_equal"] = same_shape and not out["cell_diffs"]
    if not out["bytes_equal"]:
        notes = []
        crlf_ref, crlf_new = b"\r\n" in ref_bytes, b"\r\n" in new_bytes
        if crlf_ref != crlf_new:
            notes.append(f"CRLF line endings: ref {crlf_ref}, new {crlf_new}")
        if ref_bytes.endswith(b"\n") != new_bytes.endswith(b"\n"):
            notes.append("trailing newline differs")
        out["byte_note"] = "; ".join(notes) or "bytes differ"
    return out


# --------------------------------------------------------------------------- #
# run summary                                                                   #
# --------------------------------------------------------------------------- #


def print_commit_listing(paths, root=ROOT, max_bytes=1 << 20):
    """Print the `git add -f` command for a stage's result files (CLAUDE.md: summaries,
    metadata and small arrays, each under 1 MB). Files that do not exist are left out;
    files of max_bytes or more are left out with a warning. Returns the command."""
    present = [p for p in paths if os.path.isfile(p)]
    big = [p for p in present if os.path.getsize(p) >= max_bytes]
    if big:
        print("WARNING: 1 MB or more, not listed for commit: "
              + ", ".join(os.path.relpath(p, root).replace(os.sep, "/") for p in big))
    cmd = git_add_command([p for p in present if p not in big], root) if len(present) > len(big) else None
    print("files to commit:")
    print(cmd or "(none)")
    return cmd


def git_add_command(paths, root=ROOT):
    """`git add -f` for result files (git-ignored under results/), relative to root,
    always with forward slashes."""
    rel = sorted({os.path.relpath(os.path.abspath(p), root).replace(os.sep, "/")
                  for p in paths})
    return "git add -f " + " ".join(shlex.quote(p) for p in rel)
