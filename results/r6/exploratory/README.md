# R6 exploratory analyses (E1–E3)

**Not pre-registered. Computed after gate G1 was decided** (3 October 2026; G1 did not
pass, 0 of 4 counting tasks, `results/r6/criterion/`). These analyses change no rule,
no criterion quantity and no criterion output. They were specified by the author after
the criterion results were seen.

Code: `experiments/r6_exploratory/run.py` with `config.yaml`. Inputs are the criterion
stage's inputs, verified the same way:
- the planner data on Drive, against `results/r6/planner/planner_obs_manifest.csv`
  (kind `data` only), with D7 (3)'s order check;
- the checkpoints, against D6's SHA-256 table read from the tag.

Coordinates are the primary reading: symlog for cartpole-swingup s1 and humanoid-run
s3, identity for the rest. Everything is float64. `meta_exploratory.json` records the
commit, the versions and every input's SHA-256.

| File | Content |
| --- | --- |
| `e1_directions.csv` | E1, one row per checkpoint and principal lens direction u_i, ordered by decreasing singular value of A (increasing width). Columns: D(u_i) = (q97.5 − q2.5)/2 of (x − μ)·u_i (numpy linear); r_eff(u_i) along u_i through z*; D/r_eff; whether (z* − μ)·u_i lies in [q2.5, q97.5] |
| `e1_summary.csv` | E1 per checkpoint: the largest ratio and its direction; the ratio along u_min; the number of directions with z* inside the data's range; the number with ratio ≥ 5 |
| `e2_where.csv` | E2: the fraction of states with w(x, μ) > w(z*, μ). Also z*'s whitened nearest-neighbour distance over all n states, and its percentile among the states' own nearest-other distances (percent of states whose distance is ≤ z*'s) |
| `e3_init_vs_trained.csv` | E3: tdmpc2's initialisation (e9f59321 `common/init.py`: `trunc_normal_(std=0.02)` with bounds ±2, bias 0; ε = 1e-5, H = 256), 1,000 draws per task with `numpy.random.default_rng(0)`. With zero bias the initial lens is degenerate at the origin, with widths √(Hε)/s_i. Columns: the 5th, 50th and 95th percentiles of r_eff(d₁)/D, of the anisotropy s_max/s_min and of the smallest width, beside the trained lens's ‖z*‖, ‖z* − μ‖, w(z*, μ), r_eff(d₁)/D and anisotropy. d₁, μ and D are the checkpoint's own (planner data). |
