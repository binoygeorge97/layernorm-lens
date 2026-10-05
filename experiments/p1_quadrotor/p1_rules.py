"""P-I's pre-registered rules (prereg/p1.md), as pure functions of the stage outputs.

Every function takes rows as written by run.py (dicts; values may be strings, as read
back from CSV) and the `predictions` section of the config, and returns plain dicts.
Cells are (arch, n_blocks) for one initialisation; a statement "holds in a cell" if it
holds in at least `min_seeds` of the cell's seeds.

Diverged runs (prereg/p1.md, stopping rules): a row with diverged = True fails every
rule it enters. For a per-seed rule the seed does not count toward the rule; for a
pooled rule (prediction 3) the rule fails; no seed is replaced.
"""

import numpy as np
from scipy import stats


def _f(v):
    return float(v)


def _b(v):
    return v is True or str(v) in ("True", "true", "1")


def _i(v):
    return int(float(v))


def diverged(r):
    return _b(r.get("diverged", False))


def _cells(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(_i(r[k]) if k in ("n_blocks", "seed") else r[k] for k in keys), []).append(r)
    return dict(sorted(out.items()))


def p1(analyse_rows, q):
    """Prediction 1 on the zero-bias models: per seed (a) ‖z* − μ‖ ≤ z_star_max and
    (b) R ≥ ratio_min (a diverged seed fails); per (arch, n_blocks) cell, holds if both
    hold in ≥ min_seeds."""
    c = q["p1"]
    zb = [r for r in analyse_rows if r["init"] == "zero_bias"]
    cells = []
    for (arch, nb), rr in _cells(zb, ("arch", "n_blocks")).items():
        dv = [diverged(r) for r in rr]
        a = [not d and _f(r["z_star_to_mean"]) <= float(c["z_star_max"]) for r, d in zip(rr, dv)]
        b = [not d and _f(r["nf_ratio"]) >= float(c["ratio_min"]) for r, d in zip(rr, dv)]
        both = [x and y for x, y in zip(a, b)]
        cells.append(dict(arch=arch, n_blocks=nb, n=len(rr), n_diverged=sum(dv), n_a=sum(a), n_b=sum(b),
                          n_pass=sum(both), holds=sum(both) >= int(q["min_seeds"]),
                          seeds_pass={_i(r["seed"]): p for r, p in zip(rr, both)},
                          seeds_diverged={_i(r["seed"]): d for r, d in zip(rr, dv)}))
    return cells


def p1_overall(p1_cells):
    """Prediction 1 holds if it holds in every zero-bias cell (author's change A4)."""
    return dict(holds=bool(p1_cells) and all(c["holds"] for c in p1_cells),
                n_cells_hold=sum(c["holds"] for c in p1_cells), n_cells=len(p1_cells))


def g2(p1_cells, q):
    """G2 (a planning gate): prediction 1 holds in ≥ min_cells of the zero-bias cells;
    P-III runs on the cells where it holds."""
    hold = [(c["arch"], c["n_blocks"]) for c in p1_cells if c["holds"]]
    return dict(passes=len(hold) >= int(q["g2"]["min_cells"]), n_cells_hold=len(hold), p3_cells=hold)


def p2(analyse_rows, q):
    """Prediction 2: (a) every PyTorch-default model's initial median r* within a factor
    `factor` of `r_star_estimate`; (b) per cell, R_torch < R_zero paired by seed in
    ≥ min_seeds seeds (a pair with a diverged run fails). Holds if (a) holds for every
    model and (b) in every cell."""
    c = q["p2"]
    est, fac = float(c["r_star_estimate"]), float(c["factor"])
    td = [r for r in analyse_rows if r["init"] == "torch_default"]
    a = [not diverged(r) and est / fac <= _f(r["r_star_init_median"]) <= est * fac for r in td]
    key = lambda r: (r["arch"], _i(r["n_blocks"]), _i(r["seed"]))  # noqa: E731
    by = {(r["init"],) + key(r): r for r in analyse_rows}
    cells = []
    for (arch, nb), rr in _cells(td, ("arch", "n_blocks")).items():
        wins = []
        for r in rr:
            t, z = by[("torch_default",) + key(r)], by[("zero_bias",) + key(r)]
            wins.append(not diverged(t) and not diverged(z) and _f(t["nf_ratio"]) < _f(z["nf_ratio"]))
        cells.append(dict(arch=arch, n_blocks=nb, n=len(rr), n_pass=sum(wins), holds=sum(wins) >= int(q["min_seeds"])))
    return dict(a_n=len(a), a_n_pass=sum(a), a_holds=all(a) and len(a) > 0, b_cells=cells,
                holds=all(a) and len(a) > 0 and all(x["holds"] for x in cells))


def _spearman(rows, score, target):
    x = np.array([_f(r[score]) for r in rows])
    y = np.array([_f(r[target]) for r in rows])
    return float(stats.spearmanr(x, y).statistic), len(x), int(len(x) - len(np.unique(x))), int(len(y) - len(np.unique(y)))


def p3(analyse_rows, q):
    """Prediction 3: over all models, Spearman's rank correlation between the score
    (sharpness D/r_eff along u_min) and the affected-data fraction is ≥ rho_min (a
    positive correlation; ties get average ranks, scipy.stats.spearmanr). Fails if any
    model diverged (computed over the others and reported). Also reported without a
    rule: the same correlation within each initialisation (n = 20 each)."""
    c = q["p3"]
    ok = [r for r in analyse_rows if not diverged(r)]
    rho, n, tx, ty = _spearman(ok, c["score"], c["target"])
    within = {}
    for init in sorted({r["init"] for r in ok}):
        rr = [r for r in ok if r["init"] == init]
        within[init] = dict(zip(("rho", "n", "n_tied_score", "n_tied_target"), _spearman(rr, c["score"], c["target"])))
    n_div = len(analyse_rows) - len(ok)
    return dict(n=n, n_diverged=n_div, rho=rho, holds=bool(rho >= float(c["rho_min"]) and n_div == 0),
                n_tied_score=tx, n_tied_target=ty, within_init=within)


def p4(analyse_rows, q):
    """Prediction 4 (author's changes A2, A3): on the NormedLinear 3-block models (the
    stack), per initialisation cell, the unit-free ratio P_out/P_1 along u_min is below
    ratio_max in ≥ min_seeds seeds (a diverged seed fails); holds if both cells hold. The
    point prediction: the median over the zero-bias NormedLinear 3-block models lies within
    a factor `factor` of `point` (reported as consistent or not). The pre-norm 3-block
    models and the head-on-block-1 ratio are reported without a rule."""
    c = q["p4"]
    st = [r for r in analyse_rows if _i(r["n_blocks"]) > 1]

    def cell_rows(arch):
        out = []
        for (a, init), rr in _cells([r for r in st if r["arch"] == arch], ("arch", "init")).items():
            ok = [not diverged(r) and _f(r[c["key"]]) < float(c["ratio_max"]) for r in rr]
            vals = [_f(r[c["key"]]) for r in rr if not diverged(r)]
            out.append(dict(arch=a, init=init, n=len(rr), n_pass=sum(ok), holds=sum(ok) >= int(q["min_seeds"]),
                            median=float(np.median(vals)) if vals else float("nan")))
        return out
    cells = cell_rows(c["arch"])
    zb = [_f(r[c["key"]]) for r in st if r["arch"] == c["arch"] and r["init"] == "zero_bias" and not diverged(r)]
    med = float(np.median(zb)) if zb else float("nan")
    pt, fac = float(c["point"]), float(c["factor"])
    reported = cell_rows("prenorm") if c["arch"] != "prenorm" else []
    head = {}
    for (a, init), rr in _cells(st, ("arch", "init")).items():
        v = [_f(r[c["key_head"]]) for r in rr if not diverged(r)]
        head[f"{a}/{init}"] = float(np.median(v)) if v else float("nan")
    return dict(cells=cells, holds=bool(cells) and all(x["holds"] for x in cells), zero_bias_median=med,
                n_point=len(zb), point_consistent=bool(pt / fac <= med <= pt * fac),
                reported_prenorm_cells=reported, reported_head_ratio_medians=head)


def p5(long_cells, es_cells):
    """Prediction 5: holds if it holds for every architecture of the long-budget subset
    (each cell: ≥ min_seeds of 5 pass, run.p5_cells; a diverged run fails). The
    early-stopping version per zero-bias cell is reported."""
    return dict(holds=bool(long_cells) and all(_b(c["holds"]) for c in long_cells),
                long_cells=long_cells, es_cells=es_cells)


def h1(hover_rows, q):
    """H1 on the zero-bias models: per (arch, n_blocks) cell, h1_fail (wrong in sign on
    the mask, or in stability, at hover) in ≥ min_seeds seeds; a diverged seed does not
    count. Holds if every zero-bias cell holds."""
    zb = [r for r in hover_rows if r["init"] == "zero_bias"]
    cells = []
    for (arch, nb), rr in _cells(zb, ("arch", "n_blocks")).items():
        f = [not diverged(r) and _b(r["h1_fail"]) for r in rr]
        cells.append(dict(arch=arch, n_blocks=nb, n=len(rr), n_pass=sum(f), n_diverged=sum(diverged(r) for r in rr),
                          n_no_stabilising_gain=sum(not diverged(r) and _b(r.get("no_stabilising_gain", False))
                                                    for r in rr),
                          holds=sum(f) >= int(q["min_seeds"])))
    return dict(cells=cells, holds=bool(cells) and all(c["holds"] for c in cells))


def race(p1_cells, p5_es_runs):
    """The race (reported, not tested): per zero-bias cell, seeds counted by (prediction
    1 passes, prediction 5's early-stopping version passes); diverged seeds separately."""
    p5s = {(r["arch"], _i(r["n_blocks"]), _i(r["seed"])): _b(r["passes"]) for r in p5_es_runs
           if r["init"] == "zero_bias"}
    out = []
    for c in p1_cells:
        t = {"p1_and_p5": 0, "p1_only": 0, "p5_only": 0, "neither": 0, "diverged": 0}
        for seed, a in c["seeds_pass"].items():
            if c["seeds_diverged"].get(seed):
                t["diverged"] += 1
                continue
            b = p5s.get((c["arch"], c["n_blocks"], seed))
            if b is None:
                continue
            t["p1_and_p5" if a and b else "p1_only" if a else "p5_only" if b else "neither"] += 1
        out.append(dict(arch=c["arch"], n_blocks=c["n_blocks"], **t))
    return out
