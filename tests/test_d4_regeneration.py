"""Checks of the D4 regeneration tooling that need no checkpoints (provenance.py,
regenerate_d4.py). The run itself is exercised on Colab."""

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R6 = os.path.join(ROOT, "experiments", "r6_tdmpc2")
sys.path.insert(0, R6)
import provenance as pv  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(R6, "config.yaml")))


def _has_tag(tag):
    return pv.git("cat-file", "-t", f"refs/tags/{tag}")[1] == "tag"


# --------------------------------------------------------------------------- #
# D6 SHA-256 table                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_tag(pv.D6_TAG), reason="tag prereg-r6-d3 not fetched")
def test_d6_table_from_tag_covers_every_checkpoint():
    table = pv.d6_sha_table()
    want = {pv.checkpoint_name(t, s) for t in CFG["tasks"] for s in CFG["seeds"]}
    assert set(table) == want and len(table) == 15
    # agrees with the hashes the CPU run recorded when it downloaded the files
    rec = json.load(open(os.path.join(ROOT, "results", "r6", "meta_extract.json")))["checkpoints"]
    for key, r in rec.items():
        task, seed = key.split("/")
        assert table[pv.checkpoint_name(task, seed)] == r["sha256"]


def test_parse_sha_table_rows_and_duplicates():
    a, b = "a" * 64, "b" * 64
    text = (f"| SHA-256 | File |\n| --- | --- |\n| {a} | dmcontrol/cartpole-swingup-1.pt (*) |\n"
            f"| {b} | dmcontrol/dog-run-3.pt |\nnot a row | {a} |\n")
    assert pv.parse_sha_table(text) == {"dmcontrol/cartpole-swingup-1.pt": a,
                                        "dmcontrol/dog-run-3.pt": b}
    with pytest.raises(pv.ProvenanceError):
        pv.parse_sha_table(text + f"| {b} | dmcontrol/dog-run-3.pt |\n")


def test_verify_sha256_refuses_wrong_hash(tmp_path):
    p = tmp_path / "x.pt"
    p.write_bytes(b"checkpoint bytes")
    good = pv.sha256(p)
    assert pv.verify_sha256(p, good) == good
    with pytest.raises(pv.ProvenanceError, match="refusing"):
        pv.verify_sha256(p, "0" * 64)


# --------------------------------------------------------------------------- #
# CSV comparison (the pass rule)                                                #
# --------------------------------------------------------------------------- #

REF = b"task,seed,return_mean\ncartpole-swingup,1,171.234\ncheetah-run,1,287.5\n"


def test_compare_csv_identical():
    c = pv.compare_csv(REF, REF)
    assert c["values_equal"] and c["bytes_equal"] and not c["cell_diffs"]


def test_compare_csv_detects_last_printed_digit():
    c = pv.compare_csv(REF, REF.replace(b"171.234", b"171.235"))
    assert not c["values_equal"]
    assert c["cell_diffs"] == [(0, "return_mean", "171.234", "171.235")]


def test_compare_csv_crlf_is_a_byte_difference_only():
    c = pv.compare_csv(REF, REF.replace(b"\n", b"\r\n"))
    assert c["values_equal"] and not c["bytes_equal"] and "CRLF" in c["byte_note"]


def test_compare_csv_header_and_rows():
    assert not pv.compare_csv(REF, REF.replace(b"return_mean", b"ret"))["values_equal"]
    assert not pv.compare_csv(REF, REF + b"walker-run,1,1.0\n")["values_equal"]
    assert not pv.compare_csv(REF, REF.replace(b",287.5", b",287.5,x"))["values_equal"]


def test_compare_csv_committed_results_against_themselves():
    for name in CFG["d4_regeneration"]["compare"]["pass"]:
        b = open(os.path.join(ROOT, "results", "r6", name), "rb").read()
        assert pv.compare_csv(b, b)["values_equal"]


# --------------------------------------------------------------------------- #
# manifest, summary, config                                                     #
# --------------------------------------------------------------------------- #


def test_manifest_roundtrip(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "b.npz").write_bytes(b"bb")
    (d / "a.npz").write_bytes(b"a")
    m = tmp_path / "manifest.csv"
    rows = pv.write_manifest(m, [d / "b.npz", d / "a.npz"], d)
    assert [r["file"] for r in rows] == ["a.npz", "b.npz"]
    back = pv.read_manifest(m)
    assert back[0] == dict(file="a.npz", bytes="1", sha256=pv.sha256(d / "a.npz"))


def test_git_add_command():
    cmd = pv.git_add_command([os.path.join(ROOT, "results/r6/b.csv"),
                              os.path.join(ROOT, "results/r6/a b.json")])
    assert cmd == "git add -f 'results/r6/a b.json' results/r6/b.csv"


def test_regeneration_config_is_consistent():
    rc = CFG["d4_regeneration"]
    assert set(rc["stages"]) == set(rc["expected_exit"])
    run_cfg = yaml.safe_load(pv.git("show", f"{rc['run_commit']}:experiments/r6_tdmpc2/config.yaml")[1])
    assert run_cfg["tasks"] == CFG["tasks"] and run_cfg["seeds"] == CFG["seeds"]
    assert run_cfg["data"] == CFG["data"] and run_cfg["consistency"] == CFG["consistency"]
    # the recorded run's code is the code this checkout carries (PROVENANCE.md addendum)
    for f in ("collect.py", "extract.py", "layouts.py"):
        path = f"experiments/r6_tdmpc2/{f}"
        assert pv.git("rev-parse", f"{rc['run_commit']}:{path}")[1] == pv.git("rev-parse", f"HEAD:{path}")[1]


def test_json_diffs_exact_floats():
    import regenerate_d4 as rd
    a = {"x": [1.0, 0.1 + 0.2], "y": {"z": 1}}
    assert rd._json_diffs(a, json.loads(json.dumps(a))) == []
    b = {"x": [1.0, 0.3], "y": {"z": 1, "w": 2}}
    d = rd._json_diffs(a, b)
    assert any(s.startswith("/x[1]") for s in d) and any("/y/w" in s for s in d)


def test_dm_control_install_record(monkeypatch):
    import importlib.metadata as md
    import regenerate_d4 as rd
    declared = ["absl-py>=0.7.0", "dm-tree!=0.1.2", "labmaze", "mujoco>=3.14.0",
                'h5py; extra == "hdf5"']
    installed = {"dm_control": "1.0.47", "absl-py": "2.1.0", "dm-tree": "0.1.8",
                 "mujoco": "3.14.0"}

    def version(name):
        if name not in installed:
            raise md.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(rd.md, "requires", lambda name: declared)
    monkeypatch.setattr(rd.md, "version", version)
    r = rd.dm_control_install(["labmaze"])
    assert r["missing"] == ["labmaze"] and r["unexpected_missing"] == []
    assert r["labmaze_installed"] is False and "h5py" not in r["declared_dependencies"]
    del installed["mujoco"]
    assert rd.dm_control_install(["labmaze"])["unexpected_missing"] == ["mujoco"]
