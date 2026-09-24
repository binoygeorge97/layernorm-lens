"""lens/core.py must reproduce tests/golden/*.npz.

Default mode is bit-exact and only meaningful in the environment recorded in
tests/golden/manifest.json (same JAX/jaxlib/numpy/scipy, CPU): it is a guard
against code changes, and it fails if the versions differ.

LENS_TOL=1 compares with rtol=1e-12 (atol = 1e-12 x the array's largest
magnitude, so entries that are rounding noise around zero do not fail) and skips
the full-array hashes. Use it to check other machines.
"""

import json
import os

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from tests.golden import cases

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
TOL = os.environ.get("LENS_TOL", "0") == "1"
RTOL = 1e-12


def _manifest():
    with open(os.path.join(GOLDEN, "manifest.json")) as f:
        return json.load(f)


def test_versions_match_golden():
    want, have = _manifest()["versions"], cases.versions()
    if TOL:
        pytest.skip(f"tolerance mode; golden made with {want}, running {have}")
    diff = {k: (want[k], have.get(k)) for k in want if want[k] != have.get(k)}
    assert not diff, (f"environment differs from the golden files (golden, here): {diff}. "
                      "Bit-exact comparison is not meaningful; run with LENS_TOL=1.")


@pytest.mark.parametrize("name", list(cases.CASES))
def test_case(name):
    gold = dict(np.load(os.path.join(GOLDEN, f"{name}.npz")))
    new = cases.compute(name)
    assert sorted(new) == sorted(gold), "set of outputs changed"
    bad = []
    for key, g in gold.items():
        n = np.asarray(new[key])
        if TOL and key.endswith("#sha256"):
            continue
        if g.dtype.kind in "fc" and TOL:
            scale = float(np.nanmax(np.abs(g))) if g.size and np.isfinite(g).any() else 0.0
            ok = (g.shape == n.shape and
                  np.allclose(n, g, rtol=RTOL, atol=RTOL * scale, equal_nan=True))
        elif g.dtype.kind in "fc":
            ok = g.shape == n.shape and g.dtype == n.dtype and np.array_equal(n, g, equal_nan=True)
        else:
            ok = g.shape == n.shape and np.array_equal(n, g)
        if not ok:
            bad.append(key)
    assert not bad, f"{len(bad)} outputs differ from golden ({'rtol' if TOL else 'bit-exact'}): {bad[:10]}"
