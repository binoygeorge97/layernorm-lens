"""lens/core.py must reproduce tests/golden/*.npz.

Default mode is bit-exact and only meaningful in the environment recorded in
tests/golden/manifest.json (same JAX/jaxlib/numpy/scipy, CPU): it is a guard
against code changes, and it fails if the versions differ.

LENS_TOL=1 compares with rtol=1e-12 (atol = 1e-12 x the array's largest finite
magnitude, so entries that are rounding noise around zero do not fail) and skips
the full-array hashes. Use it to check other machines. In that mode only, outputs
that are themselves rounding-noise diagnostics also pass when both values are at
noise level: *_max_abs, *_relerr, *_orth if both |golden| and |new| <= 1e-12;
*_sf (= -log10 of a relerr) if both are >= 12. *_relerr outputs also pass within
an absolute 1e-14: they are relative errors, so rtol against their own small values
(e.g. hwhm_ratio_relerr ~ 6e-7, which differs by ~2.2e-16 across platforms) is
stricter than the quantity warrants.
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
NOISE_ABS = 1e-12  # *_max_abs, *_relerr, *_orth: both at or below this
NOISE_SF = 12.0    # *_sf: both at or above this (significant figures)
RELERR_ATOL = 1e-14  # *_relerr: absolute tolerance (in addition to rtol)


def _tol_close(key, g, n):
    """Tolerance-mode comparison of one float output (LENS_TOL=1 only)."""
    if g.shape != n.shape:
        return False
    fin = np.abs(g[np.isfinite(g)])
    scale = float(fin.max()) if fin.size else 0.0  # all-NaN/inf golden: no atol
    if np.allclose(n, g, rtol=RTOL, atol=RTOL * scale, equal_nan=True):
        return True
    if key.endswith("_relerr") and np.allclose(n, g, rtol=0.0, atol=RELERR_ATOL, equal_nan=True):
        return True
    with np.errstate(invalid="ignore"):
        if key.endswith(("_max_abs", "_relerr", "_orth")):
            return bool(np.all(np.abs(g) <= NOISE_ABS) and np.all(np.abs(n) <= NOISE_ABS))
        if key.endswith("_sf"):
            return bool(np.all(g >= NOISE_SF) and np.all(n >= NOISE_SF))
    return False


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
            ok = _tol_close(key, g, n)
        elif g.dtype.kind in "fc":
            ok = g.shape == n.shape and g.dtype == n.dtype and np.array_equal(n, g, equal_nan=True)
        else:
            ok = g.shape == n.shape and np.array_equal(n, g)
        if not ok:
            bad.append(key)
    assert not bad, f"{len(bad)} outputs differ from golden ({'rtol' if TOL else 'bit-exact'}): {bad[:10]}"
