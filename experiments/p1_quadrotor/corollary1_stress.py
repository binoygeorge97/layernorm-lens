"""Corollary 1 stress test (author's item 2, 5 Oct 2026): does the analysis's Corollary 1
check stay within its 1e-10 tolerance across κ? No quadrotor data.

    python experiments/p1_quadrotor/corollary1_stress.py

For each architecture, a randomly initialised surrogate (H = 128, k = 16, ε = 1e-5,
float64; seed 0 for prenorm, 1 for normedlinear) has its first-layer bias replaced by
b = s·v.
- Sweep: v is a fixed random direction with a component outside range(PA). s is chosen
  so that κ = Hε/‖c⊥‖² takes 30 values log-uniform on [1e-4, 1e10].
- Near the tolerance: v = w + δ·n̂, with w = P A a (inside range(PA)), n̂ the unit
  out-of-range direction, and δ such that ‖c⊥‖ = r·1e-12·‖b‖ for r in {1.5, 3, 10}
  (just above geometry.DEGENERATE_RTOL), at ‖b‖ = 1.
- For each bias, synthetic standardised inputs (20,000 i.i.d. U(−√3, √3)¹⁶, z-scored)
  are shifted so their mean μ lies near z* (‖μ − z*‖ = 0.005·r_eff(u_min), so within
  0.01 r_eff along every direction) or far from it (3 standardised units, outside the
  data).
- Then run.corollary1_check, the analysis's own check, runs with the real config: lines
  through μ and through 10 training states, along d₁; on each, the maximum over the grid
  of ‖ĥ − ĥ_Cor1‖₂/‖ĥ_Cor1‖₂; the maximum over the lines.

Writes results/p1/corollary1_stress/stress.csv and meta_stress.json.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "experiments", "r6_tdmpc2")]

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import p1_load  # noqa: E402

prun = p1_load.run()
from lens import analysis as an  # noqa: E402
from lens import geometry as geo  # noqa: E402
from lens import models  # noqa: E402

SEEDS = {"prenorm": 0, "normedlinear": 1}


def _inputs(n=20000, k=16, seed=12345):
    Z = np.random.default_rng(seed).uniform(-np.sqrt(3.0), np.sqrt(3.0), (n, k))
    return (Z - Z.mean(0)) / Z.std(0)


def biases(E, eps, n_kappa=30, kappa_range=(1e-4, 1e10), near_ratios=(1.5, 3.0, 10.0), seed=7):
    """[(family, target, b)]: the κ sweep and the near-tolerance biases (module docstring)."""
    H, k = E.shape
    rng = np.random.default_rng(seed)
    P = geo.centring_projector(H)
    A = P @ E
    v = rng.standard_normal(H)
    cp = geo.lens(E, v, eps).c_perp                     # v's component outside range(PA), centred
    out = []
    for kap in np.logspace(np.log10(kappa_range[0]), np.log10(kappa_range[1]), n_kappa):
        s = np.sqrt(H * eps / (kap * float(cp @ cp)))
        out.append(("sweep", float(kap), s * v))
    n_hat = cp / np.linalg.norm(cp)
    w = A @ rng.standard_normal(k)
    w /= np.linalg.norm(w)
    for r in near_ratios:
        delta = r * geo.DEGENERATE_RTOL                 # ‖b‖ ≈ 1, so ‖c⊥‖ ≈ r·1e-12·‖b‖
        b = w + delta * n_hat
        out.append(("near_tolerance", float(r), b / np.linalg.norm(b)))
    return out


def stress(cfg, n_kappa=30):
    rows = []
    Z0 = _inputs()
    dirs = np.random.default_rng(99).standard_normal((2, 16))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for arch, seed in SEEDS.items():
        spec = models.SurrogateSpec(arch=arch, n_blocks=1, H=128, k=16, eps=1e-5, init="torch_default")
        E, _ = (np.asarray(m) for m in models.first_layer(spec, models.init_params(spec, seed)))
        for family, target, b in biases(E, spec.eps, n_kappa):
            L = geo.lens(E, b, spec.eps)
            base = dict(arch=arch, family=family, target=target, kappa=float(L.kappa),
                        norm_c_perp=float(L.norm_c_perp), norm_b=float(np.linalg.norm(b)),
                        c_perp_over_b=float(L.norm_c_perp / np.linalg.norm(b)), degenerate=bool(L.degenerate))
            r_eff_min = float(L.principal_widths_eff.min())
            for place, offset in (("near", 0.005 * r_eff_min * dirs[0]), ("far", 3.0 * dirs[1])):
                Z = Z0 + L.z_star + offset
                d1, _ = an.data_d1(Z)
                row = dict(base, mu=place, mu_to_z_star=float(np.linalg.norm(Z.mean(0) - L.z_star)),
                           r_eff_min=r_eff_min)
                if L.degenerate:  # the analysis does not check a degenerate lens
                    row.update(max_rel_dev=float("nan"), checked=False)
                else:
                    row.update(max_rel_dev=prun.corollary1_check(cfg, L, E, b, Z, d1), checked=True)
                rows.append(row)
    return rows


def main():
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    rows = stress(cfg)
    out = os.path.join(ROOT, "results", "p1", "corollary1_stress")
    prun.write_csv(os.path.join(out, "stress.csv"), rows)
    checked = [r for r in rows if r["checked"]]
    worst = max(checked, key=lambda r: r["max_rel_dev"])
    tol = float(cfg["analysis"]["corollary1"]["tol"])
    summary = dict(n_rows=len(rows), n_checked=len(checked), max_rel_dev=worst["max_rel_dev"], worst=worst,
                   tol=tol, all_within_tol=bool(worst["max_rel_dev"] <= tol))
    prun.write_json(os.path.join(out, "meta_stress.json"), prun.meta(cfg, "corollary1_stress", dict(summary=summary)))
    for r in rows:
        print(f"{r['arch']:13s} {r['family']:15s} {r['target']:9.3g} kappa {r['kappa']:9.3g} "
              f"|c|/|b| {r['c_perp_over_b']:9.3g} {r['mu']:4s} dev {r['max_rel_dev']:9.3g}")
    print(summary["all_within_tol"], summary["max_rel_dev"])


if __name__ == "__main__":
    main()
