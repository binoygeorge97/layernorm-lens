"""The `git add -f` listing printed by planner_check.py (task (d)) and the helper behind
it. extract.py and collect.py stay byte-identical to the recorded CPU run's commit
4c3f129 (test_d4_regeneration.py), so they print no listing."""

import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R6 = os.path.join(ROOT, "experiments", "r6_tdmpc2")
sys.path.insert(0, R6)
import planner_check  # noqa: E402
import provenance as pv  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(R6, "config.yaml"), encoding="utf-8"))


def _tracked(paths):
    rel = [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in paths]
    out = subprocess.run(["git", "ls-files", "--", *rel], capture_output=True, text=True,
                         encoding="utf-8", cwd=ROOT).stdout.split()
    return sorted(rel), sorted(out)


def test_planner_check_listing():
    paths = planner_check.stage_outputs(CFG)
    assert [os.path.basename(p) for p in paths] == ["planner_check.csv", "meta_planner_check.json"]
    rel, tracked = _tracked(paths)
    assert rel == tracked


def test_print_commit_listing(tmp_path, capsys):
    (tmp_path / "results" / "r6").mkdir(parents=True)
    small = tmp_path / "results" / "r6" / "a b.csv"
    small.write_text("x\n")
    big = tmp_path / "results" / "r6" / "big.npz"
    big.write_bytes(b"0" * 2048)
    missing = tmp_path / "results" / "r6" / "missing.json"
    cmd = pv.print_commit_listing([str(small), str(big), str(missing)], root=str(tmp_path), max_bytes=1024)
    out = capsys.readouterr().out
    assert cmd == "git add -f 'results/r6/a b.csv'"
    assert "WARNING" in out and "results/r6/big.npz" in out and "missing" not in out
    assert pv.print_commit_listing([str(missing)], root=str(tmp_path)) is None
