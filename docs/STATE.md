# Project state (handoff), 4 October 2026, updated after the P-I revision queue

For a fresh Claude Code session with no other context. Read this, then `CLAUDE.md`
(rules; they override defaults), `docs/theory.md` (definitions; authoritative) and
`docs/plan.md` (what each experiment must produce). Pre-registrations are in
`prereg/`. This file is a summary: where it and a tagged file disagree, the tagged
file wins.

## 1. The project in brief

Research code for a TMLR paper (submission target 17 Nov 2026; main-text
experiments frozen 1 Nov). Claim: LayerNorm after an affine map is a gnomonic
projection (theory.md, Theorem 1). Along any input line, the normalised output
depends on position only through an angle θ = arctan((s − s*)/r*), so every slope
through the layer carries a Lorentzian factor r*/(r*² + (s − s*)²) (Corollary 1).
That gives a "lens": a Cauchy-shaped zone of amplified slopes centred at z*, with
widths r* computable from the weights alone (theory.md, Diagnostics and conventions
1–6). Learned dynamics models use LayerNorm early (e.g. TD-MPC2's NormedLinear).

The argument has two halves:

- **Presence (R6):** the lens exists in a deployed model. Test: is the lens of
  TD-MPC2's first encoder layer sharp, and inside the states the robot actually
  visits? Pre-registered in `prereg/r6.md`; decision gate G1 (4 Oct 2026).
- **Harm (quadrotor, P-I / hover check / P-III / P-II):** in quadrotor
  surrogates, Jacobian error concentrates near the lens, which damages
  linearisation (LQR) and gradient-based MPC; fixes are compared at equal accuracy.
  Predictions: the author's protocol amendments v3, item 3
  (`docs/protocol_amendments_v3.md`), operationalised in `prereg/drafts/p1-draft.md`
  (v2), which becomes `prereg/p1.md` once agreed; gate G2 (18 Oct).

Gate G3 (13 Nov): every main-text claim backed.

## 2. Repository layout

| Path | What it is |
| --- | --- |
| `CLAUDE.md` | Rules: float64 everywhere; `lens/core.py` frozen; tagged prereg files never edited; cloud sessions cannot push tags; outcome metrics only after the prereg tag; `git add -f` for small results only; plan first for multi-file tasks |
| `docs/theory.md` | Definitions, theorems, implementation conventions 1–6 (confirmed 24 Sep) |
| `docs/plan.md` | Experiment plan v3: tiers, gates, per-experiment requirements |
| `docs/protocol_amendments_v3.md` | The author's protocol amendments v3, verbatim (d6b822b): surrogate target and units, the R6 criterion, P-I predictions 1–4 with numbers, display rule, P-III failure, P-I/P-II/hover/P-III amendments, claims ledger. They win over plan.md where they differ |
| `lens/core.py` | The original kink_core.py. **Frozen**: change only if `tests/test_core_regression.py` still passes bit for bit |
| `lens/geometry.py` | The lens from weights (numpy float64): `lens(E, b, eps)` → z*, c⊥, κ, principal widths/directions, Σ, degenerate flag; `line()` (r*_ℓ and r_eff along a line), `lens_distance` (ρ), `gnomonic` (Theorem 1) |
| `lens/__init__.py` | Imports core (and therefore JAX) |
| `plants/quadrotor.py` | The P-I quadrotor (JAX float64, 12 states, 4 rotor thrusts; approved cf2x.urdf parameters; X-mixer; `f`, `rk4_step`, `hover_equilibrium`, `linearize`, `linearize_step`) |
| `control/lqr.py` | Continuous and discrete LQR, closed-loop eigenvalues, spectral abscissa and radius |
| `lens/models.py`, `lens/train.py`, `lens/analysis.py` | P-I surrogates (pre-norm residual block; NormedLinear stack; zero-bias and PyTorch-default inits), training with early stopping and lens logging, and analysis (Jacobian error against lens distance, coverage and sharpness, Corollary 1 on trained models, attenuation) |
| `experiments/p1_quadrotor/` | P-I: `config.yaml` (minibatch Adam, batch 2,048; the long-budget subset), `p1_data.py` (sampling, target, standardisation, Jacobians, manifest), `p1_hover.py` (hover linearisation check), `run.py`: stages generate, train, train_long, gather, analyse, analyse_long, hover, p5, all gated on `prereg-p1`; `--index`/`--out-root` for one run per Slurm array task with a per-run JSON and SHA-256 manifest; ungated: benchmark (random targets) and init_lens (initialisation alone) |
| `docs/DECISIONS.md`, `docs/plans/` | The decision log for choices the specs leave open, and per-task plans |
| `experiments/initial_lens/` | Initial-lens check: `run.py`, `config.yaml`. 1,000 draws per (H, k, ε, init) for inits (a) zero bias, (b) torch default, (c) Flax, (d) TD-MPC2. Results in `results/initial_lens/` |
| `experiments/r6_tdmpc2/` | R6 (below) |
| `prereg/` | `r6.md` and deviations parts 1–3 (all tagged) |
| `tests/` | pytest suite; `tests/golden/` holds core.py regression outputs |
| `figures/` | The only source of paper figures (empty so far) |
| `notebooks/` | Exploration only |
| `slurm/` | P-I fallback on TACC Stampede3: `p1_array.slurm` (one array task per run), `p1_gather.slurm`, `README.md` (setup, data copy and verification, submission) |
| `results/`, `data/`, `checkpoints/` | Git-ignored. Small summaries are committed with `git add -f`; `.gitattributes` has `results/** -text` so result files are stored byte for byte |

### `experiments/r6_tdmpc2/`

| File | What it does |
| --- | --- |
| `config.yaml` | Tasks (cartpole-swingup, cheetah-run, walker-run, humanoid-run, dog-run), seeds 1–3, tdmpc2 commit e9f59321, HF revision 8fb2a82, layouts survey (D5), consistency (D3), planner_check, d4_regeneration, planner_collect and criterion sections |
| `layouts.py` | The two checkpoint layouts (D5): `detect_layout`, `check_first_layer`, numpy float64 networks `build_public` / `build_prerelease` / `build_networks`, D3 input candidates `input_candidates` (identity, symlog, LayerNorm without affine) |
| `extract.py` | Stages `list`, `extract`, `lens`: list and download checkpoints, save first-layer E, b, γ, β, ε to `results/r6/weights/`, lens to `results/r6/lens/` and `lens_summary.csv`. Requires tag prereg-r6 |
| `collect.py` | Stages `collect` (D4 policy-prior data, CPU) and `consistency` (D3 latent-consistency test, plus D5 calibration). `consistency_errors()` is D3's computation |
| `planner_check.py` | Step-1 feasibility: public `TDMPC2` class with planning on a GPU, 5 episodes, eval_mode True/False |
| `regenerate_d4.py` | Reruns the recorded CPU run (commit 4c3f129) in a git worktree to regenerate the lost D4 observations; compares with the committed CSVs |
| `provenance.py` | Standard library only: `require_prereg` (annotated tags + file match), `d6_sha_table` (read from the tag), `sha256`, `download_verified`, `write_manifest`, `compare_csv`, `git_add_command` |
| `planner_lib.py` | D6 key remap (`PRERELEASE_REMAP`, `remap_keys`), `public_shapes`, planner input (symlog), encoder agreement gate (`random_states`, `relative_errors`, `gate_rule`, `control_outcomes`, `halts`), d₁ and worst-state lens distances |
| `planner_collect.py` | D6 (a), (b), (e) collector: 15 checkpoints through tdmpc2's `TDMPC2`, controls and gate, 50 episodes each, Drive copy, manifests, `git add -f` summary. `--smoke` for 1 episode |
| `criterion_lib.py`, `criterion.py` | Criterion stage (task (c)): pure rules in `criterion_lib.py`, inputs, the six steps and outputs in `criterion.py` |
| `r6_colab.ipynb` | Original CPU pipeline notebook (extract, collect, consistency) |
| `r6_planner_check.ipynb`, `r6_d4_regeneration.ipynb`, `r6_planner_collect.ipynb` | Colab runners for the corresponding scripts |

### Entry points

```
python experiments/initial_lens/run.py --config experiments/initial_lens/config.yaml
python experiments/r6_tdmpc2/extract.py --config experiments/r6_tdmpc2/config.yaml {list,extract,lens}
python experiments/r6_tdmpc2/collect.py --config experiments/r6_tdmpc2/config.yaml {collect,consistency}
python experiments/r6_tdmpc2/planner_check.py --config experiments/r6_tdmpc2/config.yaml        # GPU, tdmpc2 env
python experiments/r6_tdmpc2/regenerate_d4.py ...                                               # see its docstring
python experiments/r6_tdmpc2/planner_collect.py --config experiments/r6_tdmpc2/config.yaml --drive DIR [--smoke]   # GPU, tdmpc2 env
python experiments/r6_tdmpc2/criterion.py --config experiments/r6_tdmpc2/config.yaml --drive DIR   # laptop, float64
python experiments/p1_quadrotor/run.py --config experiments/p1_quadrotor/config.yaml {benchmark,init_lens}   # ungated, no quadrotor data
python experiments/p1_quadrotor/run.py --config experiments/p1_quadrotor/config.yaml STAGE [--index I] [--out-root DIR] [--of train|train_long]
    # STAGE in generate, train, train_long, gather, analyse, analyse_long, hover, p5: all refuse without the tag prereg-p1
sbatch --array=0-39 slurm/p1_array.slurm train     # TACC fallback; see slurm/README.md
```

### Tests

`pytest -q` runs everything. On a machine other than the golden one, run the core
regression in tolerance mode: `LENS_TOL=1 python -m pytest -q tests/test_core_regression.py`.

| Test | Checks |
| --- | --- |
| `test_core_regression.py` | `lens/core.py` against `tests/golden/*.npz`, bit-exact in the golden environment (`tests/golden/manifest.json`); `LENS_TOL=1` for rtol 1e-12 elsewhere |
| `test_lens.py` | `lens/geometry.py`, Theorem 1 |
| `test_d4_regeneration.py` | `regenerate_d4.py` logic |
| `test_planner_collect.py` | remap, gate rule, controls, halting, manifests (no checkpoints or GPU needed) |
| `test_criterion.py` | criterion stage, synthetic arrays only |
| `test_quadrotor.py` | hover equilibrium, Jacobians against central differences, signs, the hover linearisation, RK4's order |
| `test_p1_models.py`, `test_p1_pipeline.py` | P-I surrogates, training (minibatches, u_min logging, fixed budgets), analysis, LQR, hover check, data and stages (long-budget runs, snapshots, gather and its SHA-256 refusals, prediction 5's rule, init_lens), on synthetic plants only; the gated stages refuse without `prereg-p1` |

## 3. Branches, merge policy, tags

- **Working branches:** `r6-criterion` (R6; to be merged into `main` by the author's PR)
  and `quadrotor-sim` (session 3 onwards, created from `r6-criterion` at 732aa44).
  Never commit on `main`. Own working branches may be pushed after each completed
  queue item (CLAUDE.md).
- `main` is at 31be62e (PR #8, which merged `claude/new-session-0r0qe0`). It
  contains everything up to this handoff, including the planner results (3d52d3d)
  and the WIP criterion stage (eb8835a). Changes reach `main` only through pull
  requests that the author merges.
- `claude/new-session-0r0qe0` and `claude/loving-johnson-wk9yb3` (earlier
  sessions) are fully merged into `main`. Do not develop on them. Each Colab
  notebook's first cell is a `BRANCH` parameter, set to `r6-criterion`.
- Policy as practised: a feature branch with pull requests into `main`, merged by
  the author. Use merge commits, never rebase or force-push shared branches, and
  never rewrite history. Commit and push only what was asked. A cloud session
  cannot push tags: commit and push the prereg file, then the author creates and
  pushes the annotated tag. Every script for a pre-registered test refuses to run
  unless its annotated tags exist and the prereg files match them.

### Pre-registration tags (all annotated, all on origin)

| Tag | Tag object | Commit | File | Settles |
| --- | --- | --- | --- | --- |
| `prereg-r6` | 086fbdd5c2aef676a6bfe438f9fcda714c153274 | 66b3a09f09a91e525e7348b065b52d7907b2929b | `prereg/r6.md` | Question, models (seed 1 counts), data (50 episodes, env seeds 0–4), definitions, the criterion (Inside, Populated, Sharp), G1 (≥ 2 of the 4 non-dog tasks), what is reported regardless |
| `prereg-r6-d1` | 3a0bb105b2b42223a1cafa9912514ad5246edafd | e5060cf9813401cd652088186641ad913320baa8 | `prereg/r6-deviations.md` | D1 which keys are the layer (pre-release layout); D2 ε = 1e-5 assumed; D3 encoder position 0 decided by a latent-consistency test (identity, symlog, LayerNorm), with an accept rule e ≤ 0.1·e₀ and lowest; D4 data from the policy prior, since the planner could not load the files |
| `prereg-r6-d2` | df32edce53bcf0c541a285acc1383eb15c1ac4da | e20f5a61c98956da46bbb1d1bf5c6ac5aa4db817 | `prereg/r6-deviations-2.md` | D5 two checkpoint layouts: pre-release for cartpole-swingup s1 and humanoid-run s3, public for the other 13; D1–D3 apply only to the pre-release ones; D3 calibration on public seed-1 checkpoints; layout detection rule |
| `prereg-r6-d3` | 3dec0649dddb1a2d2125bd22b1a40afd2124beef | bc0b23b52baa14ff35f67d3b752c61b4da9a72c9 | `prereg/r6-deviations-3.md` | D6 planner data (public tdmpc2 planner, eval_mode=True, float32 for acting only); symlog at pre-release position 0; key remap; encoder agreement gate and controls; conditions (i) ≥ 0.9 × published and (ii) for cartpole s1; humanoid s3 labels; (c) calibration; (e) protocol; (g) end-to-end susceptibility (reported only); the 15-checkpoint SHA-256 table |
| `prereg-r6-d4` | b87cac0e2c604ee3a879fd39567db8b32ac64d6e | 66bd7174ec0202051cce41ae6d47e10e0d7f4e48 | `prereg/r6-deviations-4.md` | D7 clarifications before the criterion stage: D3's tie rule under D6 (every reading within 10% must pass); e₀ each candidate's own; Populated's row order and nearest-neighbour set; Corollary 1 along lines through μ and data states; all three readings for the pre-release files; D6 (g) median state sets; vector signs; laptop environment with a Colab rerun if any statistic is within 1% of its threshold |

### Key commits since the tags (on `claude/new-session-0r0qe0`)

| Commit | What |
| --- | --- |
| 4bc1a14 | `results/r6/prerelease_keys.json`: all keys and shapes of the two pre-release checkpoints |
| 00603a8 | CPU-run outputs and metadata recovered from the Drive copy of 24 Sep (`meta_*.json`, `lens/`, `weights/`, `lens_summary.csv`, `consistency.json`); never committed before because `results/` is ignored |
| 132bb72 | `results/r6/PROVENANCE.md` addendum: the CPU run is recorded at 4c3f129 (not c85262f; the R6 code is identical) |
| fa6d02d | CLAUDE.md: commit summaries, metadata and arrays < 1 MB; observation data only via SHA-256 manifest |
| 2ac54e2, fb46698, e739a71 | D4 regeneration: script, then outputs. **MATCH**: the regenerated run reproduces the 4c3f129 CPU run byte for byte (`results/r6/d4_regen/`, `meta_d4_regeneration.json` match: true, `d4_obs_manifest.csv`) |
| 2f9764c, f4187dc | `.gitattributes` `results/** -text`; `lens_summary.csv` restored to its original (CRLF) bytes |
| 98bb91e | PROVENANCE.md addendum: D4 observations regenerated, byte for byte |
| 54a7288, bca1e71 | D6 planner collector, tests and notebook; notebook streams output live |
| 3d52d3d | D6 planner results, all 15 checkpoints (30 files; see open task (a)) |
| eb8835a | WIP criterion-stage draft (open task (c)) |

### Commits on `r6-criterion`

| Commit | What |
| --- | --- |
| 2a44aea, 6abc1e2 | CLAUDE.md laptop and git-safety sections; UTF-8 git output and portable paths |
| 66bd717 | `prereg/r6-deviations-4.md` (D7), tagged `prereg-r6-d4` |
| 11843a3 | Criterion stage, reviewed (task (c)) |
| 8331c12 | `results/r6/planner/PROVENANCE-2026-10-03.md` (task (a)) |
| 8bcba37 | `criterion.check_lens`: z* compared norm-wise (the first run stopped in input verification on dog-run s1's z*: laptop vs Colab differ by 7.7e-14 norm-wise, cond(A) ≈ 1e4; nothing had been computed) |
| 57e7689 | `results/r6/criterion/`: the criterion stage outputs (45 files) |
| 1360cb4, 55f1358 | Notebook `BRANCH` parameter (task (f)); `git add -f` listings (task (d)) |
| 6d503b7, 2e2d312 | CLAUDE.md: stop and ask on any failed check; test the committed state after every commit. 2e2d312 reverts 55f1358's `collect.py`/`extract.py` listings (they must stay byte-identical to 4c3f129) |
| 56bfd75 | `results/r6/meta_list.json` committed, with a `PROVENANCE.md` addendum on its origin |
| 7249b68, d3e197f | Exploratory analyses E1–E3 (not pre-registered): code, then outputs and tables in `results/r6/exploratory/README.md` |

### Commits on `quadrotor-sim` (from 732aa44 on `r6-criterion`)

| Commit | What |
| --- | --- |
| 85bdb74 | `plants/quadrotor.py` and its tests (queue item 2) |
| 5d8b917, 08d1deb | P-I infrastructure (queue item 3); module renames so the suite collects |
| 3c1289a | `results/p1/benchmark/`: full-batch runtime on random targets |
| def116d | `prereg/drafts/p1-draft.md` v1 |
| d6b822b | `docs/protocol_amendments_v3.md`, verbatim |
| d7b2cb3 | Minibatch default, u_min lens log, `train_long`, `gather`, `analyse_long`, `p5`, Slurm scripts |
| 74058c0 | `results/p1/benchmark_bs2048/`: minibatch runtime on random targets |
| 1b7ee27, 07b21b4 | `init_lens` stage; `results/p1/init_lens/` |

## 4. R6 status

**G1 failed (0 of 4) on 3 October 2026.** Per r6.md, "the headline becomes 'whether the
defect appears depends on a default'". **The R6 criterion stage has run** (`results/r6/criterion/`, commit 57e7689; run at
8bcba37 on the laptop, status `complete`; nothing within 1% of a threshold, so no
Colab rerun under D7 (8)). Results, by the rules of r6.md and D3–D7:

- **Corollary 1** (D7 (4), D6 (g)): 225 lines, maximum relative deviation 3.94e-14.
- **cartpole-swingup s1** (D6 (b)): condition (i) 1.002 ≥ 0.9; condition (ii) on the
  planner data: symlog lowest, e/e₀ = 3.69e-5 ≤ 0.1 (identity 0.105, LayerNorm
  0.111); identified. D7 (1) ratios e_c/e_symlog: identity 2903, LayerNorm 2984;
  not ambiguous. Record-only D4 value: symlog e/e₀ 0.0204, lowest (reproduces
  `consistency.csv`).
- **humanoid-run s3**: label "confirmed" (condition (i) 1.058; symlog lowest, e/e₀
  0.0068).
- **Criterion:** no checkpoint meets all three of Inside, Populated and Sharp, under
  any reading. Sharp fails for all 15 under the primary reading (D/r_eff(d₁)
  0.85–4.08); Populated fails for all 19 checkpoint-readings; Inside holds only for
  cheetah-run s1 and s2 (and cartpole s1 under the reported identity reading).
- **G1: does not pass**, 0 of 4 counting tasks (cartpole-swingup s1, cheetah-run s1,
  walker-run s1, humanoid-run s1; none flagged). r6.md: "If not, the headline
  becomes 'whether the defect appears depends on a default'."
- D6 (c) calibration, D6 (g) susceptibility, ρ fractions: in `results/r6/criterion/`.
- Exploratory analyses after G1 (not pre-registered): `results/r6/exploratory/` (E1
  directional sharpness, E2 where z* sits, E3 initialisation versus trained). Its
  README also tabulates the reported-regardless ρ and ρ_eff fractions.

Pipeline results and quantities D6 itself recorded, from before the stage:

- **Lens from the weights** (`results/r6/lens_summary.csv`, run at 4c3f129): all 15
  checkpoints are non-degenerate, κ ≤ 0.00165. The r* principal widths are strongly
  anisotropic: width_max / width_min is 4.1–5.0 (cartpole-swingup), 93–142
  (cheetah-run), 71–74 (walker-run), 28–172 (humanoid-run) and 9,040–11,000
  (dog-run). dog-run ‖z*‖ is 563–769. The pre-release lenses are in symlog
  coordinates (the coordinates the layer receives under D6 (b)).
- **D3 on the D4 policy-prior data** (`results/r6/consistency.csv`): identity was
  rejected for both pre-release checkpoints. cartpole-swingup s1 e/e₀: identity
  0.036, symlog 0.020 (lowest), LayerNorm 0.046. humanoid-run s3: identity 0.989,
  symlog 0.115 (lowest), LayerNorm 0.965. Public calibration (identity):
  cheetah-run 0.014, walker-run 0.013, humanoid-run s1 0.117. This led to D6.
- **D4 regeneration:** the lost D4 observations were regenerated by rerunning
  4c3f129; `returns.csv` and `consistency.csv` were reproduced byte for byte. The
  observations are on Drive, verified by `results/r6/d4_obs_manifest.csv`.
- **Encoder agreement gate and controls** (`results/r6/planner/encoder_gate.csv`;
  identical values in all four sessions, data and smoke): the positive control
  (cartpole s2) passed, max relative error 2.3e-7 (random) and 4.5e-7 (D4 states).
  The negative control (cartpole s1 read with identity) failed as required, max
  1.35 and 1.25 (> 1e-2). The gate passed for cartpole s1 (median 2.4e-7 / 3.5e-7,
  max 4.9e-7 / 6.8e-7) and humanoid s3 (median 2.2e-7 / 3.0e-7, max 6.5e-7 /
  5.2e-7), against the rule median ≤ 1e-5 and max ≤ 1e-3.
- **Planner returns** (`results/r6/planner/planner_returns.csv`; 50 episodes,
  eval_mode=True, Tesla T4), as fractions of the published return:

  | Task | Seed 1 | Seed 2 | Seed 3 |
  | --- | --- | --- | --- |
  | cartpole-swingup | 1.002 (symlog) | 1.001 | 1.000 |
  | cheetah-run | 1.004 | 0.987 | 1.010 |
  | walker-run | 1.025 | 1.009 | 1.008 |
  | humanoid-run | 1.233 | 1.147 | 1.058 (symlog) |
  | dog-run | 0.916 | 0.883 | 1.008 |

  No checkpoint is flagged below 0.5. Cartpole s2 was collected in session
  20260925T214748Z; the other 14 in session 20260927T141627Z.
- **D6 (b) condition (i) passed** for cartpole-swingup s1 (1.002 ≥ 0.9) and
  humanoid-run s3 (1.058).

## 5. Environments and data

| Environment | Where | Versions | Used for |
| --- | --- | --- | --- |
| CPU pipeline | Colab CPU | Python 3.13.15, jax/jaxlib 0.10.2, numpy 2.4.6, scipy 1.17.1, torch 2.14.0+cpu, mujoco 3.14.0, dm_control 1.0.47 (installed without labmaze for the regeneration), PyYAML 6.0.1 (`meta_d4_regeneration.json`) | extract, collect (D4), consistency (D3), D4 regeneration; the D7 (8) rerun of the criterion stage if anything is borderline |
| Planner | Colab Tesla T4, CUDA 12.6 | tdmpc2 e9f59321's pinned env (`docker/environment.yaml`) in a Python 3.11.16 uv virtualenv: torch 2.7.1+cu126, tensordict 0.8.3, torchrl 0.8.1, mujoco 3.1.2, dm_control 1.0.16, numpy 1.24.4, gymnasium 0.29.1, hydra-core 1.3.2, omegaconf 2.3.0 (planner metas) | planner_check, planner_collect. No JAX; our float64 work there is numpy |
| Laptop | Windows, VS Code + Claude Code, `.venv/` (git-ignored) | Recorded 3 Oct 2026, see section 7. Install `requirements.txt` (jax 0.10.2, numpy 2.4.6, scipy 1.17.1, PyYAML 6.0.1, pytest 9.1.1, matplotlib 3.11.2) and, for R6 work, `requirements-r6.txt` (torch 2.14.0 CPU first, from its own index). Record the versions on first use. The core regression is bit-exact only in the golden environment, so use `LENS_TOL=1` | development, tests, the criterion stage (D7 (8)) |

Source code: tdmpc2 is cloned to `checkpoints/tdmpc2_src` at
e9f59321933cbc8e11a002b842adc7d4ffae8ff1. Checkpoints come from
huggingface.co/nicklashansen/tdmpc2 at revision
8fb2a82efb3bae96941da440128fe1332e4394fd, into `checkpoints/tdmpc2/dmcontrol/`,
each verified against D6's SHA-256 table (read from the tag).

**Google Drive, `MyDrive/layernorm-lens-r6/`:**

| Drive path | Contents | Verified by |
| --- | --- | --- |
| `data/r6/<task>-seed<s>.npz` | D4 policy-prior observations (15 files), regenerated | `results/r6/d4_obs_manifest.csv`, plus `meta_d4_regeneration.json` with match: true |
| `data/r6/planner/<task>-seed<s>.npz` | D6 planner data (15 files): obs (50, 501, k) float32, actions (50, 500, a), rewards, env_seed, episode_in_seed, seconds | `results/r6/planner/planner_obs_manifest.csv` (kind `data`) and each `meta_<task>-seed<s>.json` |
| `data/r6/planner_smoke/` | 2 smoke episodes (outcome data, never used for any rule) | `planner_obs_manifest.csv` (kind `smoke`) |
| `results/r6/planner/`, `results/r6/planner_smoke/` | Copies of the result files | the committed copies |
| (24 Sep copy) | The original CPU run's results, from which 00603a8 was recovered | `results/r6/PROVENANCE.md` |

## 6. Open tasks, in order

**(a) Provenance note.** Done: `results/r6/planner/PROVENANCE-2026-10-03.md` (8331c12).
Both planner sessions ran bca1e71; 5 checkpoints planned compiled and 10 eagerly
(from walker-run s1 on); the missing smoke key mapping is a deterministic function
of the keys and was regenerated byte for byte.

**(b) Drive copy check.** Every stage must check that each file in its
`git add -f` list exists on Drive, with the same SHA-256, before it reports
success. The criterion stage does this for its own outputs; `planner_collect.py` and
`regenerate_d4.py` do not yet. This is the cause of the missing smoke key mapping
(confirmed in (a)): `planner_collect.load_agent` writes a pre-release checkpoint's key mapping
when it loads it, which happens for the gate, but `run_checkpoint` pushes it to
Drive only for a checkpoint that acts (`planner_collect.py:472-473`). In a smoke
run only cartpole-swingup s1 acts, so humanoid-run s3's smoke mapping was written
locally and never copied. Its `planner/` copy reached Drive because humanoid-run
s3 acted in the full run.

**(c) Criterion stage.** Done: run at 8bcba37, outputs in 57e7689 (section 4). Code reviewed and approved by the author on 3 Oct 2026,
after the seven questions were settled by D7 (`prereg/r6-deviations-4.md`, tag
`prereg-r6-d4`). `criterion.py` runs on the laptop in float64 (D7 (8)). It refuses
to run unless the five annotated tags are present locally and on origin, the
prereg files match them, and the working tree is clean. It verifies every planner
observation file (kind `data` rows only) and every D4 file against the committed
manifests on Drive, with D7 (3)'s order check. It then runs, in order: (1) the
Corollary 1 checks, D7 (4) and D6 (g), stopping on any line above 1e-10; (2) D6 (b)
condition (ii) with D7 (2), the record-only D4 value, the humanoid s3 label and the
D6 (c) calibration; (3) Inside, Populated and Sharp for all 15 checkpoints, under
all three readings for the pre-release files (D7 (5)); (4) the D7 (1) tie rule, the
counting checkpoints and G1; (5) D6 (g), reported only; (6) the D7 (8) borderline
scan, which marks the stage "pending Colab rerun" if anything lies within 1% of its
threshold. Outputs go to `results/r6/criterion/`, are copied to Drive and verified
there, and the stage prints the `git add -f` list. Tests: `tests/test_criterion.py`
(synthetic only).

**(d)** Done for `planner_check.py`, which prints its `git add -f` listing
(`provenance.print_commit_listing`; `tests/test_stage_listings.py`). `extract.py` and
`collect.py` print none: they must stay byte-identical to the recorded CPU run's
commit 4c3f129, the guarantee `results/r6/PROVENANCE.md` rests on
(`test_d4_regeneration.py`). Their listing, added in 55f1358, was reverted.

**(e) Session 3: the quadrotor simulator.** Done on `quadrotor-sim`: `plants/quadrotor.py`
with the approved parameters (m 0.027 kg, L 0.0397 m X configuration, J diag(1.4e-5,
1.4e-5, 2.17e-5) kg·m², kf 3.16e-10, km 7.94e-12, c = km/kf, thrust-to-weight 2.25,
f_max = 2.25 m g/4, g = 9.81; gym-pybullet-drones uses 9.8), from gym-pybullet-drones'
`cf2x.urdf` (889ce4a5c068ae4d811df1442ceb4f4d6cdf43eb, SHA-256 81494018…884b). Linear
drag is behind a flag with no approved coefficients. Choices: `docs/DECISIONS.md`.

**(e2) P-I infrastructure.** Built and tested on synthetic problems only (see the layout
table, `docs/plans/p1-infrastructure.md` and `docs/plans/p1-revision.md`). No surrogate
is trained on quadrotor data, and no lens quantity is computed on a trained quadrotor
surrogate, before the tag `prereg-p1`.
- Built: data generation, surrogates, training with lens logging, analysis, the hover
  linearisation check.
- Training default (author, 4 Oct): minibatch Adam, batch 2,048; held-out MSE on the
  full validation set every 500 steps; early stopping unchanged.
- The long-budget subset: `train_long`, 10 runs at 500k steps, with snapshots.
- Array runs with per-run manifests, and `gather`.
- Benchmarks on random targets:
  - full batch (`results/p1/benchmark/`, 08d1deb): 0.06–0.27 s per step, at most
    172 h for the grid;
  - batch 2,048 (`results/p1/benchmark_bs2048/`, d7b2cb3): 0.016–0.042 s per step,
    at most 30.2 h for the 40-model grid and 24.5 h for the long-budget subset, lens
    logging included.
- Initial lens values from initialisation alone: `results/p1/init_lens/` (1b7ee27).
- TACC fallback: `slurm/` (Stampede3 arrays; queue, allocation and wall time marked
  EDIT).
- Still to write, after the author's D16: the hover-check trims and H1's cell rule.

**(g) `prereg/p1.md`.** Draft v2 committed: `prereg/drafts/p1-draft.md` (not a prereg
file), for the author and advisor. Next: the author's decisions, then `prereg/p1.md`
committed alone and tagged `prereg-p1` by the author. Only then may any P-I stage touch
quadrotor data.
- Predictions 1–4 are quoted exactly from the amendments' item 3, each with a proposed
  operationalisation.
- Prediction 5 (zero bias): along u_min, r_eff/D rises at least 10× and reaches at
  least 0.1, with κ ≤ 0.1 at the end, in at least 4 of 5 seeds per architecture.
  - Primary test: the long-budget subset.
  - Also evaluated on the 40 runs' trajectories up to early stopping, as the race with
    prediction 1.
  - The initial zero-bias value is 0.0087.
- The hover check and H1 are included, and G2 is proposed.
- 18 decisions (D1–D18), each with a recommendation.

**(h) Console log of the main planner session.** Done:
`results/r6/planner/logs/console_20260927T141627Z.txt` is the author's transcription of
the Colab cell-7 output. It was verified episode by episode against the stored rewards
(700 returns, 14 means; no mismatch). The provenance note's 4 Oct addendum cites it for
the recompile_limit (8) warning (14:51:36 UTC, walker-run s1).

**(f) Colab notebooks' `BRANCH`.** Done: each of the four notebooks now opens with a
marked `BRANCH` parameter cell, set to `r6-criterion`.

## Questions for the author

Non-blocking; work continues on everything that does not depend on the answers.

Answered on 4 Oct 2026:
- linear drag: off for P-I, a modelling choice in the draft;
- compute: minibatch Adam, with TACC Slurm arrays as the fallback;
- predictions 1–4: `docs/protocol_amendments_v3.md`.

Open:

1. The 18 decisions (D1–D18) at the end of `prereg/drafts/p1-draft.md`, each with a
   recommendation. Most consequential:
   - D7: whether prediction 1 needs both the centre clause and the ratio clause;
   - D9: the affected-data fraction;
   - D11 and D12: prediction 5's primary test and the 500k-step budget;
   - D15: G2;
   - D16: H1 and the trims.
2. Prediction 4 quotes "E1b's median ratio of about 0.2", but E1b's definition of that
   ratio is not in the repository. The draft proposes output-over-block-1 sharpness
   along u_min. What did E1b measure?
3. TACC, only if the fallback is used: which system, queue and allocation? The scripts
   assume Stampede3 and mark these lines EDIT.
4. For the record: the amendments' preamble says their items are committed "before G1
   on 4 October". The file was committed on 4 Oct (d6b822b), after G1 was evaluated on
   3 Oct.
   - Item 2's criterion was already tagged in `prereg/r6.md` (prereg-r6) before any R6
     data.
   - Item 3's predictions will be fixed by `prereg-p1` before any quadrotor data.

   Nothing was therefore fixed after seeing a result, but the paper's provenance note
   may want to say so.

## 7. Known issues

- The core regression's bit-exact golden test depends on the platform. On the
  laptop, 5 cases (stack, geometry, evaluate, data, train) fail bit for bit,
  although jax, jaxlib, numpy and scipy are the versions in
  `tests/golden/manifest.json`. With `LENS_TOL=1` (rtol 1e-12) the file passes: 8
  passed, 1 skipped.
- torch._dynamo hit its recompile limit (8, a guard on `kwargs['t0']`) during
  walker-run s1, so the planner ran eagerly from walker-run s1 on: walker s1–s3,
  humanoid, dog and both pre-release runs. The runs therefore differ from D6 (e)'s
  wording (task (a)2). GPU execution is not bitwise deterministic anyway; the
  stored observations are the data of record.
- Colab's GitHub token is read-only, so results are committed from the laptop.
  Result files must keep their bytes: either `core.autocrlf=false`, or rely on
  `.gitattributes` (`results/** -text`). The laptop's system gitconfig
  (`C:/Program Files/Git/etc/gitconfig`) sets `core.autocrlf=true`; every tracked
  file under `results/` is marked `-text` by `.gitattributes`, so git leaves those
  bytes alone.
- The Drive copy gap: `planner_collect` copies a pre-release key mapping to Drive
  only for a checkpoint that acts, and nothing checks the Drive copies against the
  `git add -f` list. That is why `results/r6/planner_smoke/key_mapping_humanoid-run-seed3.json`
  is missing from 3d52d3d (30 files, not 31). See tasks (a)4 and (b).
- Laptop environment (3 Oct 2026; Windows 11 Pro 10.0.26200): Python 3.12.1,
  jax/jaxlib 0.10.2 (CPU backend), numpy 2.4.6, scipy 1.17.1, PyYAML 6.0.1,
  pytest 9.1.1, matplotlib 3.11.2, torch 2.14.0+cpu. Every `requirements.txt` pin
  matches. Of `requirements-r6.txt`, torch matches, but dm_control (1.0.47) and
  mujoco (3.14.0) are not installed. Python differs from the Colab CPU runtime
  (3.13.15).
- Tests on the laptop at 31be62e: 9 failed and 93 passed with plain `pytest -q`.
  - 5 are the golden cases above.
  - 2 (`test_d6_table_from_tag_covers_every_checkpoint`, `test_remap_table_is_d6s`)
    failed because `provenance.git()` ran `subprocess.run(..., text=True)` without
    an encoding. Windows decoded git's UTF-8 output as cp1252, so `require_prereg`
    and `d6_sha_table` also failed on the laptop.
  - 2 (`test_git_add_command`, `test_verify_d4`) failed on path separators:
    `git_add_command` printed `results\r6\...`, and `verify_d4`'s Drive path mixed
    separators.
  - Fixed on `r6-criterion`: git output is decoded as UTF-8, `git_add_command`
    always prints forward slashes, and `verify_d4` builds native paths. With
    `LENS_TOL=1` and without `PYTHONUTF8`: 105 passed, 1 skipped.
- The interrupted planner session left no session file (task (a)1).
- ε = 1e-5 is an assumption for the pre-release checkpoints (D2). κ, ‖c⊥‖ and r*
  are reported, so any ε-dependent quantity can be recomputed.
- dog-run (k = 223) is close to the k ≤ 255 limit and does not count toward G1. Its
  z* lies far outside the data (‖z*‖ 563–769).
- humanoid-run's calibration e/e₀ (0.117 for a correctly read public network) is
  above D3's 0.1 threshold, which is why D6 does not apply 0.1 to humanoid-run.
- The D4 regeneration installed dm_control without labmaze (no Python 3.13 wheel);
  only `dm_control.locomotion` needs it. The planner env records dm_control 1.0.16.
- `results/r6/lens_summary.csv` has CRLF line endings, the original bytes. Keep
  `.gitattributes` (`results/** -text`); on Windows, do not let editors or
  `core.autocrlf` rewrite result files.
- `results/r6/PROVENANCE.md`'s original text names c85262f; its addendums
  correct this. Do not rewrite the original text; add addendums.
