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

Run at 7249b68 (clean tree), 3 Oct 2026, 8 min 29 s, on the laptop; versions in
`meta_exploratory.json`. Values below are rounded; the CSVs hold full precision.

## E1. Directional sharpness (exploratory)

u_i are numbered by decreasing singular value of A, so u_0 is the narrowest direction
(u_min). "Inside" means (z* − μ)·u lies within the data's [q2.5, q97.5] along u.

| Checkpoint | Reading | k | Max D/r_eff | At u_i (r_eff, D) | z* inside there | D/r_eff along u_min | z* inside along u_min | Directions with z* inside | Directions with D/r_eff ≥ 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole-swingup s1 | symlog | 5 | 1.85 | u_0 (0.281, 0.519) | yes | 1.85 | yes | 5 of 5 | 0 |
| cartpole-swingup s2 | identity | 5 | 0.839 | u_4 (4.6, 3.86) | yes | 0.346 | yes | 5 of 5 | 0 |
| cartpole-swingup s3 | identity | 5 | 0.776 | u_4 (4.99, 3.87) | yes | 0.325 | yes | 5 of 5 | 0 |
| cheetah-run s1 | identity | 17 | 1.53 | u_14 (12.3, 18.9) | yes | 0.648 | yes | 17 of 17 | 0 |
| cheetah-run s2 | identity | 17 | 1.40 | u_4 (0.707, 0.988) | yes | 0.309 | yes | 17 of 17 | 0 |
| cheetah-run s3 | identity | 17 | 1.54 | u_11 (7.59, 11.7) | yes | 0.256 | yes | 17 of 17 | 0 |
| walker-run s1 | identity | 24 | 5.00 | u_18 (3.21, 16.0) | yes | 0.931 | yes | 21 of 24 | 1 |
| walker-run s2 | identity | 24 | 5.26 | u_18 (3.30, 17.3) | yes | 0.795 | yes | 22 of 24 | 1 |
| walker-run s3 | identity | 24 | 4.78 | u_18 (3.40, 16.3) | yes | 0.862 | yes | 21 of 24 | 0 |
| humanoid-run s1 | identity | 67 | 2.46 | u_55 (16.7, 41.2) | yes | 0.677 | no | 59 of 67 | 0 |
| humanoid-run s2 | identity | 67 | 2.20 | u_57 (19.0, 41.9) | yes | 0.688 | yes | 61 of 67 | 0 |
| humanoid-run s3 | symlog | 67 | 0.758 | u_52 (6.31, 4.78) | yes | 0.146 | no | 61 of 67 | 0 |
| dog-run s1 | identity | 223 | 2.08 | u_4 (0.260, 0.541) | yes | 0.936 | no | 184 of 223 | 0 |
| dog-run s2 | identity | 223 | 1.69 | u_100 (2.36, 3.99) | yes | 0.699 | no | 199 of 223 | 0 |
| dog-run s3 | identity | 223 | 1.82 | u_54 (0.872, 1.59) | yes | 0.336 | yes | 175 of 223 | 0 |

## E2. Where z* sits (exploratory)

Whitened distances use C⁺ of all n = 25,050 states. The nearest-neighbour (NN)
distances are over all n states. The percentile is the percent of states whose own
nearest-other distance is ≤ z*'s NN distance.

| Checkpoint | Reading | w(z*, μ) | Fraction of states with w(x, μ) > w(z*, μ) | z* NN distance | Data NN median | Data NN 95th pct | z* NN percentile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole-swingup s1 | symlog | 5.771 | 0.0322 | 1.45 | 0.00576 | 0.0296 | 100.00 |
| cartpole-swingup s2 | identity | 5.699 | 0.0367 | 4.13 | 0.00424 | 0.0559 | 100.00 |
| cartpole-swingup s3 | identity | 7.099 | 0.0054 | 4.75 | 0.0073 | 0.0626 | 100.00 |
| cheetah-run s1 | identity | 4.833 | 0.1444 | 2.73 | 0.749 | 2.14 | 96.93 |
| cheetah-run s2 | identity | 3.550 | 0.4075 | 2.59 | 1.02 | 1.95 | 98.16 |
| cheetah-run s3 | identity | 6.003 | 0.0842 | 3.84 | 1.16 | 1.82 | 99.79 |
| walker-run s1 | identity | 16.99 | 0.0139 | 13.3 | 0.480 | 2.13 | 99.98 |
| walker-run s2 | identity | 16.09 | 0.0168 | 12.9 | 0.486 | 1.87 | 99.92 |
| walker-run s3 | identity | 16.55 | 0.0169 | 13.8 | 0.493 | 1.63 | 99.98 |
| humanoid-run s1 | identity | 52.38 | 0.0000 | 41.6 | 3.67 | 9.40 | 100.00 |
| humanoid-run s2 | identity | 46.49 | 0.0000 | 41.2 | 4.02 | 7.47 | 100.00 |
| humanoid-run s3 | symlog | 52.26 | 0.0000 | 45.0 | 3.31 | 11.6 | 100.00 |
| dog-run s1 | identity | 3624 | 0.0000 | 3610 | 12.6 | 17.9 | 100.00 |
| dog-run s2 | identity | 876 | 0.0000 | 863 | 12.7 | 17.6 | 100.00 |
| dog-run s3 | identity | 10010 | 0.0000 | 9980 | 11.1 | 21.6 | 100.00 |

## E3. Initialisation versus trained (exploratory)

The initial lens is degenerate at the origin (z* = 0), so its ‖z* − μ‖ is ‖μ‖ and its
w(z*, μ) is w(0, μ). Init columns give p5 / p50 / p95 over 1,000 draws. The
anisotropy is s_max/s_min. The widths are r* for the trained lens and the ε-limited
√(Hε)/s_max for the initial one.

| Checkpoint | k | ‖z*‖ trained (init) | ‖z* − μ‖ trained (init) | w(z*, μ) trained (init) | r_eff(d₁)/D trained | r_eff(d₁)/D init | Anisotropy trained | Anisotropy init | Min width trained | Min width init p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole-swingup s1 | 5 | 1.61 (0) | 1.71 (0.565) | 5.77 (3.12) | 0.496 | 0.0739 / 0.0792 / 0.0851 | 4.10 | 1.15 / 1.23 / 1.31 | 0.281 | 0.144 |
| cartpole-swingup s2 | 5 | 0.487 (0) | 0.778 (0.858) | 5.70 (4.50) | 1.04 | 0.0387 / 0.0416 / 0.0449 | 4.68 | 1.15 / 1.23 / 1.31 | 0.982 | 0.144 |
| cartpole-swingup s3 | 5 | 1.18 (0) | 1.10 (0.859) | 7.10 (4.31) | 1.17 | 0.0388 / 0.0417 / 0.0450 | 5.02 | 1.15 / 1.23 / 1.31 | 0.995 | 0.144 |
| cheetah-run s1 | 17 | 3.75 (0) | 6.79 (8.48) | 4.83 (6.21) | 0.441 | 0.00626 / 0.00673 / 0.00727 | 93.4 | 1.51 / 1.60 / 1.70 | 0.163 | 0.129 |
| cheetah-run s2 | 17 | 3.70 (0) | 6.27 (9.43) | 3.55 (6.18) | 0.536 | 0.00544 / 0.00581 / 0.00628 | 139 | 1.51 / 1.60 / 1.70 | 0.149 | 0.129 |
| cheetah-run s3 | 17 | 3.49 (0) | 7.26 (9.70) | 6.00 (10.9) | 0.587 | 0.00563 / 0.00602 / 0.00648 | 142 | 1.51 / 1.60 / 1.70 | 0.152 | 0.129 |
| walker-run s1 | 24 | 2.69 (0) | 5.51 (7.94) | 17.0 (25.5) | 0.395 | 0.00422 / 0.00454 / 0.00492 | 74.4 | 1.69 / 1.78 / 1.90 | 0.219 | 0.124 |
| walker-run s2 | 24 | 3.00 (0) | 5.83 (7.84) | 16.1 (24.0) | 0.400 | 0.00407 / 0.00438 / 0.00475 | 70.6 | 1.69 / 1.78 / 1.90 | 0.234 | 0.124 |
| walker-run s3 | 24 | 3.23 (0) | 5.93 (7.94) | 16.5 (23.9) | 0.426 | 0.00401 / 0.00432 / 0.00468 | 73.9 | 1.69 / 1.78 / 1.90 | 0.239 | 0.124 |
| humanoid-run s1 | 67 | 6.85 (0) | 11.9 (8.68) | 52.4 (38.2) | 0.245 | 0.00206 / 0.00223 / 0.00241 | 172 | 2.79 / 2.95 / 3.13 | 0.366 | 0.107 |
| humanoid-run s2 | 67 | 5.85 (0) | 12.7 (11.8) | 46.5 (35.3) | 0.329 | 0.00200 / 0.00215 / 0.00231 | 141 | 2.79 / 2.95 / 3.13 | 0.363 | 0.107 |
| humanoid-run s3 | 67 | 1.69 (0) | 3.67 (3.31) | 52.3 (32.5) | 0.732 | 0.0185 / 0.0198 / 0.0213 | 28.3 | 2.79 / 2.95 / 3.13 | 0.779 | 0.107 |
| dog-run s1 | 223 | 563 (0) | 560 (50.2) | 3620 (32.8) | 0.291 | 0.000468 / 0.000501 / 0.000539 | 9930 | 23.4 / 26.8 / 32.3 | 0.161 | 0.0828 |
| dog-run s2 | 223 | 633 (0) | 592 (154) | 876 (24.7) | 0.308 | 0.000131 / 0.000141 / 0.000152 | 9040 | 23.4 / 26.8 / 32.3 | 0.213 | 0.0828 |
| dog-run s3 | 223 | 769 (0) | 777 (42.6) | 10000 (40.0) | 0.297 | 0.000827 / 0.000885 / 0.000953 | 11000 | 23.4 / 26.8 / 32.3 | 0.188 | 0.0828 |

## Reported-regardless R6 quantities (pre-registered, not exploratory)

r6.md, "Reported regardless of outcome": the fractions of observations with ρ_eff ≤ 1,
10, 100 and 1,000, and the same for ρ when the lens is non-degenerate (all 15 are).
These are in the primary reading, from the criterion stage's
`results/r6/criterion/rho_fractions.csv` (commit 57e7689). They are tabulated here for
convenience and were not recomputed.

| Checkpoint | Reading | ρ_eff ≤1 | ≤10 | ≤100 | ≤1000 | ρ ≤1 | ≤10 | ≤100 | ≤1000 | Median ρ_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole-swingup s1 | symlog | 0.0133 | 0.9852 | 1.0000 | 1.0000 | 0.0133 | 0.9851 | 1.0000 | 1.0000 | 5.335 |
| cartpole-swingup s2 | identity | 0.8962 | 1.0000 | 1.0000 | 1.0000 | 0.8962 | 1.0000 | 1.0000 | 1.0000 | 0.5331 |
| cartpole-swingup s3 | identity | 0.9041 | 1.0000 | 1.0000 | 1.0000 | 0.9041 | 1.0000 | 1.0000 | 1.0000 | 0.5288 |
| cheetah-run s1 | identity | 0.0039 | 0.4881 | 0.9973 | 1.0000 | 0.0039 | 0.4880 | 0.9973 | 1.0000 | 10.22 |
| cheetah-run s2 | identity | 0.0053 | 0.8369 | 0.9993 | 1.0000 | 0.0053 | 0.8368 | 0.9993 | 1.0000 | 5.675 |
| cheetah-run s3 | identity | 0.0076 | 0.8456 | 1.0000 | 1.0000 | 0.0075 | 0.8455 | 1.0000 | 1.0000 | 5.529 |
| walker-run s1 | identity | 0.0000 | 0.0000 | 0.9771 | 1.0000 | 0.0000 | 0.0000 | 0.9764 | 1.0000 | 41.16 |
| walker-run s2 | identity | 0.0000 | 0.0000 | 0.9931 | 1.0000 | 0.0000 | 0.0000 | 0.9929 | 1.0000 | 35.85 |
| walker-run s3 | identity | 0.0000 | 0.0000 | 0.9980 | 1.0000 | 0.0000 | 0.0000 | 0.9980 | 1.0000 | 33.27 |
| humanoid-run s1 | identity | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 46.97 |
| humanoid-run s2 | identity | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 49.69 |
| humanoid-run s3 | symlog | 0.0000 | 0.9999 | 1.0000 | 1.0000 | 0.0000 | 0.9999 | 1.0000 | 1.0000 | 7.354 |
| dog-run s1 | identity | 0.0000 | 0.0000 | 0.0797 | 1.0000 | 0.0000 | 0.0000 | 0.0797 | 1.0000 | 118.8 |
| dog-run s2 | identity | 0.0000 | 0.0000 | 0.9977 | 1.0000 | 0.0000 | 0.0000 | 0.9977 | 1.0000 | 81.71 |
| dog-run s3 | identity | 0.0000 | 0.0000 | 0.2870 | 1.0000 | 0.0000 | 0.0000 | 0.2870 | 1.0000 | 104.3 |
