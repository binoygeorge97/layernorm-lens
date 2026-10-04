"""R6 exploratory analyses E1-E3: NOT pre-registered, computed after G1 was decided.

    python experiments/r6_exploratory/run.py --config experiments/r6_exploratory/config.yaml \
        --drive "G:\\My Drive\\layernorm-lens-r6"

Reads the same inputs as the criterion stage, verified the same way (planner data on
Drive against planner_obs_manifest.csv, kind "data" only, with D7 (3)'s order check;
checkpoints against D6's SHA-256 table read from the tag), through criterion.py's
own functions, which it imports and does not change. Everything is float64, in the
primary reading of each checkpoint.

E1  Directional sharpness: for every principal lens direction u_i (right singular
    vectors of A, signed as D7 (7)), the data half-width D(u_i) = (q97.5 − q2.5)/2 of
    (x − μ)·u_i against r_eff(u_i) along u_i through z*; and whether (z* − μ)·u_i lies
    within [q2.5, q97.5] of the data's projections.
E2  Where z* sits: the fraction of data states x with w(x, μ) > w(z*, μ); z*'s whitened
    nearest-neighbour distance over all n states, as a percentile of the states' own
    nearest-other distances (percent of states whose distance is ≤ z*'s).
E3  Initialisation versus trained: tdmpc2's init (trunc_normal std 0.02, zero bias,
    ε = 1e-5; H = 256) for each task's k, 1,000 draws. Zero bias gives a degenerate
    lens at the origin with ε-limited widths √(Hε)/s_i. Reported beside the trained
    lens: ‖z*‖, ‖z* − μ‖ and w(z*, μ), r_eff(d₁)/D, and the anisotropy s_max/s_min.

Outputs in results/r6/exploratory/ with meta (config, commit, versions); prints
`git add -f`.
"""

import argparse
import datetime
import os
import platform
import sys

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments", "r6_tdmpc2"))
import criterion as crit  # noqa: E402  (read-only use of the criterion stage's input checks)
import criterion_lib as cl  # noqa: E402
import layouts  # noqa: E402
import provenance as pv  # noqa: E402

geo = cl.geo


# --------------------------------------------------------------------------- #
# E1-E3: pure computations                                                      #
# --------------------------------------------------------------------------- #


def e1_directions(L, X, S, q=(2.5, 97.5)):
    """One row per principal direction u_i (descending singular value, so ascending
    width): D(u_i), r_eff(u_i), their ratio, and whether z* lies within the data's
    q-range along u_i."""
    X = np.asarray(X, np.float64)
    rows = []
    for i in range(L.k):
        u = cl.canonical_sign(L.principal_dirs[:, i])
        p = (X - S.mu) @ u
        lo, hi = np.percentile(p, q)
        D = float((hi - lo) / 2.0)
        r_eff = float(geo.line(L, u)["r_eff"])
        pz = float((L.z_star - S.mu) @ u)
        rows.append(dict(i=i, singular_value=float(L.sing[i]), r_star=float(L.principal_widths[i]),
                         r_eff=r_eff, D=D, ratio=D / r_eff, z_proj=pz, q_lo=float(lo), q_hi=float(hi),
                         z_inside=bool(lo <= pz <= hi)))
    return rows


def e1_summary(rows, L):
    i_max = int(np.argmax([r["ratio"] for r in rows]))
    i_umin = int(np.argmin(L.principal_widths))
    return dict(k=len(rows), max_ratio=rows[i_max]["ratio"], max_ratio_i=i_max,
                max_ratio_r_eff=rows[i_max]["r_eff"], max_ratio_D=rows[i_max]["D"],
                max_ratio_z_inside=rows[i_max]["z_inside"],
                umin_i=i_umin, umin_ratio=rows[i_umin]["ratio"], umin_z_inside=rows[i_umin]["z_inside"],
                n_dirs_z_inside=int(sum(r["z_inside"] for r in rows)),
                n_dirs_ratio_ge_5=int(sum(r["ratio"] >= 5 for r in rows)))


def e2_where(S, X, z_star):
    """Fraction of states with w(x, μ) > w(z*, μ); z*'s whitened nearest-neighbour
    distance over all n states and its percentile among the states' own nearest-other
    distances (percent of states whose nearest-other distance is ≤ z*'s)."""
    X = np.asarray(X, np.float64)
    wz = float(cl.whitened(S, z_star, S.mu))
    wx = cl.whitened(S, X, S.mu)
    Y = (X - S.mu) @ S.whitener
    zw = (np.asarray(z_star, np.float64) - S.mu) @ S.whitener
    dz = float(np.min(np.linalg.norm(Y - zw, axis=1)))
    nn = cl.nearest_other(Y)
    return dict(n=int(len(X)), w_z_mu=wz, frac_w_x_gt_w_z=float(np.mean(wx > wz)),
                z_nn_dist=dz, data_nn_median=float(np.median(nn)), data_nn_p95=float(np.percentile(nn, 95)),
                z_nn_percentile=float(100.0 * np.mean(nn <= dz)))


def trunc_normal(rng, shape, std, a, b):
    """torch.nn.init.trunc_normal_(mean=0, std, a, b) by rejection: N(0, std²) restricted
    to [a, b] (absolute bounds, as torch)."""
    x = rng.normal(0.0, std, size=shape)
    bad = (x < a) | (x > b)
    while bad.any():
        x[bad] = rng.normal(0.0, std, size=int(bad.sum()))
        bad = (x < a) | (x > b)
    return x


def e3_init_draws(k, H, std, trunc, eps, n_draws, rng, d1):
    """n_draws initial first layers (E trunc-normal, b = 0). With b = 0, c⊥ = 0 and
    z* = 0: a degenerate lens at the origin whose ε-limited widths are √(Hε)/s_i
    (theory.md convention 4) and whose width along d₁ is √(Hε)/‖A d₁‖. Returns the
    singular values of A = P E (n_draws, k) and ‖A d₁‖ (n_draws,)."""
    P = geo.centring_projector(H)
    sings, nAd = np.empty((n_draws, k)), np.empty(n_draws)
    for j in range(n_draws):
        E = trunc_normal(rng, (H, k), std, trunc[0], trunc[1])
        if j == 0:
            L0 = geo.lens(E, np.zeros(H), eps)
            assert L0.degenerate and np.allclose(L0.z_star, 0)
        A = P @ E
        sings[j] = np.linalg.svd(A, compute_uv=False)
        nAd[j] = np.linalg.norm(A @ d1)
    return sings, nAd


def e3_row(L, S, X, sings, nAd, H, eps, qs=(5, 50, 95)):
    """Trained lens beside the initial-lens distribution, for one checkpoint."""
    D = cl.half_width_D(S, X)
    r_eff_tr = float(geo.line(L, S.d1)["r_eff"])
    init_r = np.sqrt(H * eps) / nAd / D
    init_aniso = sings[:, 0] / sings[:, -1]
    init_wmin = np.sqrt(H * eps) / sings[:, 0]
    row = dict(trained_norm_z_star=float(np.linalg.norm(L.z_star)),
               trained_dist_z_mu=float(np.linalg.norm(L.z_star - S.mu)),
               trained_w_z_mu=float(cl.whitened(S, L.z_star, S.mu)),
               trained_r_eff_d1_over_D=r_eff_tr / D,
               trained_anisotropy=float(L.sing[0] / L.sing[-1]),
               trained_width_min=float(L.principal_widths.min()),
               init_norm_z_star=0.0, init_dist_z_mu=float(np.linalg.norm(S.mu)),
               init_w_z_mu=float(cl.whitened(S, np.zeros(S.k), S.mu)), D=D)
    for name, v in (("init_r_eff_d1_over_D", init_r), ("init_anisotropy", init_aniso),
                    ("init_width_min", init_wmin)):
        for q in qs:
            row[f"{name}_p{q}"] = float(np.percentile(v, q))
    return row


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()
    state = pv.git_state()
    if state["git_dirty"]:
        sys.exit("refusing to run: the working tree has uncommitted changes to tracked files.")
    xc = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg = yaml.safe_load(open(os.path.join(ROOT, xc["criterion_config"]), encoding="utf-8"))
    try:
        table = pv.d6_sha_table()
    except pv.ProvenanceError as e:
        sys.exit(str(e))
    drive = os.path.abspath(args.drive)
    out = os.path.join(ROOT, *xc["results"].split("/"))
    os.makedirs(out, exist_ok=True)
    t0 = datetime.datetime.now(datetime.timezone.utc)
    checkpoints = [(t, s, cfg["survey"][t][s]) for t in cfg["tasks"] for s in cfg["seeds"]]
    planner, planner_rec = crit.planner_inputs(cfg, drive, checkpoints)
    ini = xc["e3_init"]
    dirs, e1s, e2s, e3s, ck_rec = [], [], [], [], {}
    for task, seed, layout in checkpoints:
        n = f"{task}-seed{seed}"
        print(f"{n} [{layout}]", flush=True)
        sd, mods, sha = crit.load_checkpoint(cfg, table, task, seed)
        ck_rec[pv.checkpoint_name(task, seed)] = sha
        lin, _ = layouts.check_first_layer(sd, layout, cfg["layer"]["layouts"], n)
        L = geo.lens(sd[f"{lin}.weight"], sd[f"{lin}.bias"], float(cfg["layer"]["layernorm_eps"]))
        O = planner[(task, seed)][0]
        reading = crit.primary_reading(layout)
        fn = layouts.input_candidates(float(cfg["consistency"]["pos0_layernorm_eps"]))[reading]
        X = fn(O.reshape(-1, O.shape[-1]))
        S = cl.data_stats(X, float(cfg["criterion"]["kept_rtol"]))
        base = dict(task=task, seed=seed, reading=reading)
        rows = e1_directions(L, X, S, tuple(xc["data_quantiles"]))
        dirs += [dict(base, **r) for r in rows]
        e1s.append(dict(base, **e1_summary(rows, L)))
        e2s.append(dict(base, **e2_where(S, X, L.z_star)))
        # the same seed for every checkpoint: the seeds of a task share the same draws
        sings, nAd = e3_init_draws(L.k, int(ini["H"]), float(ini["std"]), ini["trunc"], float(ini["eps"]),
                                   int(ini["n_draws"]), np.random.default_rng(int(ini["rng_seed"])), S.d1)
        e3s.append(dict(base, k=L.k, **e3_row(L, S, X, sings, nAd, int(ini["H"]), float(ini["eps"]),
                                                tuple(ini["summary_quantiles"]))))
    crit.write_csv(os.path.join(out, "e1_directions.csv"), dirs)
    crit.write_csv(os.path.join(out, "e1_summary.csv"), e1s)
    crit.write_csv(os.path.join(out, "e2_where.csv"), e2s)
    crit.write_csv(os.path.join(out, "e3_init_vs_trained.csv"), e3s)
    crit.write_json(os.path.join(out, "meta_exploratory.json"), dict(
        stage="r6_exploratory", pre_registered=False,
        note="Not pre-registered; computed after G1 was decided (3 Oct 2026).",
        **state, versions=crit.versions(), platform=platform.platform(),
        time_start=t0.isoformat(), time_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inputs=dict(checkpoints=ck_rec, planner=planner_rec), config=xc, criterion_config=cfg))
    files = [os.path.join(out, f) for f in ("e1_directions.csv", "e1_summary.csv", "e2_where.csv",
                                             "e3_init_vs_trained.csv", "meta_exploratory.json", "README.md")]
    pv.print_commit_listing(files)


if __name__ == "__main__":
    main()
