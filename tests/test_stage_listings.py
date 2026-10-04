"""The `git add -f` listings printed by the finished R6 stages (extract.py, collect.py,
planner_check.py; task (d)). Only their summary printing is under test: each stage's
listed files must be exactly the committed outputs of that stage."""

import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R6 = os.path.join(ROOT, "experiments", "r6_tdmpc2")
sys.path.insert(0, R6)
import collect  # noqa: E402
import extract  # noqa: E402
import planner_check  # noqa: E402
import provenance as pv  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(R6, "config.yaml"), encoding="utf-8"))


def _tracked(paths):
    rel = [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in paths]
    out = subprocess.run(["git", "ls-files", "--", *rel], capture_output=True, text=True,
                         encoding="utf-8", cwd=ROOT).stdout.split()
    return sorted(rel), sorted(out)


def test_extract_listings_are_the_committed_outputs():
    for stage, n in (("extract", 16), ("lens", 17)):
        paths = extract.stage_outputs(CFG, stage)
        rel, tracked = _tracked(paths)
        assert len(paths) == n and rel == tracked, stage
    assert any(p.endswith("lens_summary.csv") for p in extract.stage_outputs(CFG, "lens"))
    # `extract list` writes only its meta. results/r6/meta_list.json was never committed
    # (00603a8 recovered the other metas); the listing names it so the next run prints it.
    assert [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in extract.stage_outputs(CFG, "list")] == \
        ["results/r6/meta_list.json"]


def test_collect_listings_never_include_observations():
    for stage, want in (("collect", ["returns.csv", "meta_collect.json"]),
                        ("consistency", ["consistency.csv", "consistency.json", "meta_consistency.json"])):
        paths = collect.stage_outputs(CFG, stage)
        assert [os.path.basename(p) for p in paths] == want
        rel, tracked = _tracked(paths)
        assert rel == tracked
        assert not any("/data/" in r or r.startswith("data/") for r in rel)


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
