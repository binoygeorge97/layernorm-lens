"""P-I's pre-registered rules (prereg/p1.md), as pure functions of the stage outputs.

Every function takes rows as written by run.py (dicts; values may be strings, as read
back from CSV) and the `predictions` section of the config, and returns plain dicts.
Cells are (arch, n_blocks) for one initialisation; a statement "holds in a cell" if it
holds in at least `min_seeds` of the cell's seeds.
"""

import numpy as np
from scipy import stats


def _f(v):
    return float(v)


def _b(v):
    return v is True or str(v) in ("True", "true", "1")


def _i(v):
    return int(float(v))


def _cells(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(_i(r[k]) if k in ("n_blocks", "seed") else r[k] for k in keys), []).append(r)
    return dict(sorted(out.items()))


def p1(analyse_rows, q):
    """Prediction 1 on the zero-bias models: per seed (a) ‖z* − μ‖ ≤ z_star_max and
    (b) R ≥ ratio_min; per (arch, n_blocks) cell, holds if both hold in ≥ min_seeds."""
    c = q["p1"]
    zb = [r for r in analyse_rows if r["init"] == "zero_bias"]
    cells = []
    for (arch, nb), rr in _cells(zb, ("arch", "n_blocks")).items():
        a = [_f(r["z_star_to_mean"]) <= float(c["z_star_max"]) for r in rr]
        b = [_f(r["nf_ratio"]) >= float(c["ratio_min"]) for r in rr]
        both = [x and y for x, y in zip(a, b)]
        cells.append(dict(arch=arch, n_blocks=nb, n=len(rr), n_a=sum(a), n_b=sum(b), n_pass=sum(both),
                          holds=sum(both) >= int(q["min_seeds"]),
                          seeds_pass={_i(r["seed"]): p for r, p in zip(rr, both)}))
    return cells


def g2(p1_cells, q):
    """G2 (a planning gate): prediction 1 holds in ≥ min_cells of the zero-bias cells;
    P-III runs on the cells where it holds."""
    hold = [(c["arch"], c["n_blocks"]) for c in p1_cells if c["holds"]]
    return dict(passes=len(hold) >= int(q["g2"]["min_cells"]), n_cells_hold=len(hold), p3_cells=hold)


def p2(analyse_rows, q):
    """Prediction 2: (a) every PyTorch-default model's initial median r* within a factor
    `factor` of `r_star_estimate`; (b) per cell, R_torch < R_zero paired by seed in
    ≥ min_seeds seeds. Holds if (a) holds for every model and (b) in every cell."""
    c = q["p2"]
    est, fac = float(c["r_star_estimate"]), float(c["factor"])
    td = [r for r in analyse_rows if r["init"] == "torch_default"]
    a = [est / fac <= _f(r["r_star_init_median"]) <= est * fac for r in td]
    key = lambda r: (r["arch"], _i(r["n_blocks"]), _i(r["seed"]))  # noqa: E731
    R = {(r["init"],) + key(r): _f(r["nf_ratio"]) for r in analyse_rows}
    cells = []
    for (arch, nb), rr in _cells(td, ("arch", "n_blocks")).items():
        wins = [R[("torch_default",) + key(r)] < R[("zero_bias",) + key(r)] for r in rr]
        cells.append(dict(arch=arch, n_blocks=nb, n=len(rr), n_pass=sum(wins), holds=sum(wins) >= int(q["min_seeds"])))
    return dict(a_n=len(a), a_n_pass=sum(a), a_holds=all(a) and len(a) > 0, b_cells=cells,
                holds=all(a) and len(a) > 0 and all(x["holds"] for x in cells))


def p3(analyse_rows, q):
    """Prediction 3: Spearman's rank correlation over all models between the score
    (sharpness D/r_eff along u_min) and the affected-data fraction is ≥ rho_min (a
    positive correlation). Ties get average ranks (scipy.stats.spearmanr)."""
    c = q["p3"]
    x = np.array([_f(r[c["score"]]) for r in analyse_rows])
    y = np.array([_f(r[c["target"]]) for r in analyse_rows])
    rho = float(stats.spearmanr(x, y).statistic)
    return dict(n=len(x), rho=rho, holds=bool(rho >= float(c["rho_min"])),
                n_tied_score=int(len(x) - len(np.unique(x))), n_tied_target=int(len(y) - len(np.unique(y))))


def p4(analyse_rows, q):
    """Prediction 4 on the 3-block models: per (arch, init) cell, the ratio
    S_out/S_block1 along u_min is below ratio_max in ≥ min_seeds seeds (holds if every
    cell holds); the point prediction: the median over the zero-bias 3-block models lies
    within a factor `factor` of `point` (reported as consistent or not)."""
    c = q["p4"]
    st = [r for r in analyse_rows if _i(r["n_blocks"]) > 1]
    cells = []
    for (arch, init), rr in _cells(st, ("arch", "init")).items():
        att = [_f(r[c["key"]]) < float(c["ratio_max"]) for r in rr]
        cells.append(dict(arch=arch, init=init, n=len(rr), n_pass=sum(att), holds=sum(att) >= int(q["min_seeds"])))
    zb = [_f(r[c["key"]]) for r in st if r["init"] == "zero_bias"]
    med = float(np.median(zb)) if zb else float("nan")
    pt, fac = float(c["point"]), float(c["factor"])
    return dict(cells=cells, holds=bool(cells) and all(x["holds"] for x in cells), zero_bias_median=med,
                point_consistent=bool(pt / fac <= med <= pt * fac))


def p5(long_cells, es_cells):
    """Prediction 5: holds if it holds for every architecture of the long-budget subset
    (each cell: ≥ min_seeds of 5 pass, run.p5_cells). The early-stopping version per
    zero-bias cell is reported."""
    return dict(holds=bool(long_cells) and all(_b(c["holds"]) for c in long_cells),
                long_cells=long_cells, es_cells=es_cells)


def h1(hover_rows, q):
    """H1 on the zero-bias models: per (arch, n_blocks) cell, h1_fail (wrong in sign or
    stability at hover) in ≥ min_seeds seeds; holds if every zero-bias cell holds."""
    zb = [r for r in hover_rows if r["init"] == "zero_bias"]
    cells = []
    for (arch, nb), rr in _cells(zb, ("arch", "n_blocks")).items():
        f = [_b(r["h1_fail"]) for r in rr]
        cells.append(dict(arch=arch, n_blocks=nb, n=len(rr), n_pass=sum(f),
                          n_no_stabilising_gain=sum(_b(r.get("no_stabilising_gain", False)) for r in rr),
                          holds=sum(f) >= int(q["min_seeds"])))
    return dict(cells=cells, holds=bool(cells) and all(c["holds"] for c in cells))


def race(p1_cells, p5_es_runs):
    """The race (reported, not tested): per zero-bias cell, seeds counted by (prediction
    1 passes, prediction 5's early-stopping version passes)."""
    p5s = {(r["arch"], _i(r["n_blocks"]), _i(r["seed"])): _b(r["passes"]) for r in p5_es_runs
           if r["init"] == "zero_bias"}
    out = []
    for c in p1_cells:
        t = {"p1_and_p5": 0, "p1_only": 0, "p5_only": 0, "neither": 0}
        for seed, a in c["seeds_pass"].items():
            b = p5s.get((c["arch"], c["n_blocks"], seed))
            if b is None:
                continue
            t["p1_and_p5" if a and b else "p1_only" if a else "p5_only" if b else "neither"] += 1
        out.append(dict(arch=c["arch"], n_blocks=c["n_blocks"], **t))
    return out
