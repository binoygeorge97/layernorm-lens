# Pre-registration: P-I, the lens in quadrotor surrogates

Date: {COMMIT_DATE}. Author: Binoy George.

- **Tag.** This file is identified by the annotated git tag `prereg-p1`. The author also
  creates a GitHub Release from that tag, which records a server-side timestamp.
- **Deviations.** This file is never edited after tagging. Any deviation goes in a
  separate, dated file, `prereg/p1-deviations.md`.
- **Code.** The code and configuration at the tagged commit implement everything below:
  - `experiments/p1_quadrotor/`: `config.yaml`, `run.py`, `p1_data.py`, `p1_hover.py`,
    `p1_rules.py`, and the launcher `launch.py`;
  - `lens/` and `plants/quadrotor.py`.

  Every stage that touches quadrotor data refuses to run unless this tag exists and this
  file matches it. `docs/plans/p1_traceability.md` maps every rule and reported quantity
  below to its function and test.
- **Provenance of the predictions.** Predictions 1–4 are quoted from the author's
  protocol amendments v3 (`docs/protocol_amendments_v3.md`). That file was committed on
  4 October 2026 (d6b822b), after G1 was evaluated on 3 October.
  - Its R6 item had been tagged in `prereg/r6.md` (prereg-r6) before any R6 data.
  - Its P-I items precede any quadrotor data: none has been generated.

## Question

Gate G2 (18 October 2026): do zero-bias quadrotor surrogates concentrate Jacobian error
at hover, near the first-layer lens? And does training repair the lens (prediction 5)
before the early-stopping budget freezes the defect in (prediction 1)?

## Plant

- `plants/quadrotor.py` at the tagged commit.
  - States: 12, in the order position (3), velocity (3), ZYX Euler angles (roll,
    pitch, yaw) and body rates (3). Yaw is not wrapped.
  - Inputs: 4 rotor thrusts.
- Parameters: the author-approved Crazyflie 2.x values from gym-pybullet-drones'
  `cf2x.urdf` (889ce4a5c068ae4d811df1442ceb4f4d6cdf43eb), with g = 9.81 m/s².
- **No aerodynamic drag.** This is a modelling choice: no linear drag coefficient is in
  the approved set, and the surrogates learn whatever map the simulator defines.
- RK4 at dt = 0.01 s with zero-order hold. Hover thrust per rotor is u0 = m g/4;
  f_max = 2.25 u0.

## Data

- **Sampling:** i.i.d. uniform in a box around hover, with every range symmetric about
  hover. Hover is at the origin with yaw 0, at rest, with thrust u0.

  | Input | Range |
  | --- | --- |
  | position | ±1 m on each axis |
  | velocity | ±1 m/s on each axis |
  | roll, pitch | ±30° |
  | yaw | ±180° |
  | body rates | ±2 rad/s on each axis |
  | each rotor thrust | u0 (1 ± 0.5), i.e. [0.5 u0, 1.5 u0], inside [0, f_max] and never clipped |

- **Sizes:** 20,000 training, 5,000 validation and 5,000 test states, drawn in that
  order from `numpy.random.default_rng(0)`.
- **Surrogate input:** z = (x, u) ∈ R¹⁶ (k = 16).
- **Target:** the scaled increment y = (x_{t+1} − x_t)/dt ∈ R¹² (amendments item 1).
- **Standardisation:** z and y are z-scored per coordinate with the training split's
  mean and standard deviation. Because every range is symmetric, the standardised mean
  is hover in every input dimension, up to sampling noise.
- **Ground truth:** J = ∂y/∂z by forward-mode autodiff of the simulator. In standardised
  coordinates, J_std = diag(1/σ_y) J diag(σ_z).
- **Storage:** the data stays outside the repository. A SHA-256 manifest
  (`results/p1/data_manifest.csv`) is committed when the data are generated, and every
  stage verifies the data against it before use.

## Surrogates

- **Architectures:**
  - (a) Pre-norm residual block: h = E z + b. Then, per block,
    h ← h + W2 GELU(W1 (γ LN(h) + β) + b1) + b2, with branch width H. Finally
    y = Wo h + bo.
  - (b) TD-MPC2-style NormedLinear stack: Linear → LayerNorm (affine) → Mish per
    block, then a Linear head.
- **Initialisations:**
  - `zero_bias`: weights U(±1/√fan_in), every bias 0.
  - `torch_default`: weights and biases U(±1/√fan_in).
  - The two initialisations of a seed share every weight and the minibatch stream.
- **Grid:** 2 architectures × 2 initialisations × {1 block, 3 blocks} × seeds 0–4, at
  H = 128 and ε = 1e-5. That is 40 surrogates.
  - The first layer is the first draw for every architecture and depth, so a seed's
    initial lens is the same in all four cells.
- **Training:** float64 minibatch Adam.
  - Batch: 2,048 distinct training rows, drawn afresh at every step from the run's seed
    and the step.
  - lr 3e-3; (β1, β2, ε) = (0.9, 0.999, 1e-8).
  - Every 500 steps: the held-out one-step MSE on the full validation set.
  - Early stopping (amendments item 6): within a budget of 100,000 steps, patience
    10,000 steps (10% of the budget), tolerance 1%. An evaluation is an improvement if
    it is below 99% of the MSE at the last improvement.
  - The returned parameters are those with the lowest held-out MSE. They are the
    early-stopping budget's model.
- **The long-budget subset:** zero bias, 1 block, both architectures, seeds 0–4 (10
  runs).
  - Each trains for a fixed 500,000 steps with early stopping off.
  - Parameter snapshots are saved every 10,000 steps.
- **Lens log:** every 500 steps, in both training stages:
  - z*, κ, ‖c⊥‖ and the principal widths r* and r_eff;
  - along u_min and along d₁: r*, r_eff, D, r_eff/D and S = ‖J(z*) d‖.
- **Execution:** the laptop CPU (Intel Core i7-9750H, 6 cores; Windows 11), Python 3.12.1, jax/jaxlib
  0.10.2, numpy 2.4.6, scipy 1.17.1. `launch.py` runs 3 processes at a time, each with
  `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`,
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS`
  all 1, `JAX_PLATFORMS=cpu` and `PYTHONHASHSEED=0`. With this environment the outputs
  were byte-identical across repeated and concurrent runs (`results/p1/determinism/`).
  Every stage writes on local disk (`--out-root`); results reach Google Drive only through
  SHA-256-verified copies (`copy_verified.py`).

## Definitions

All lens quantities follow `docs/theory.md` at the tagged commit. The lens is the first
LayerNorm's, in the surrogate's standardised input coordinates.

- **μ:** the mean of the training inputs.
- **hover_std:** hover's own standardised coordinates, (z_hover − mean)/sd with the
  training split's mean and standard deviation.
- **d₁:** the unit eigenvector of the training inputs' covariance (numpy.cov, ddof = 1)
  with the largest eigenvalue.
- **u_min:** the lens's narrowest principal direction, the right singular vector of A
  for its largest singular value. It is recomputed from the current weights wherever it
  is used.
- **D(d):** (q97.5(p) − q2.5(p))/2 of p = (z − μ)·d over the training inputs, with
  numpy's default (linear) percentile method.
- **κ:** Hε/‖c⊥‖² (theory.md, Setting); κ = ∞ for a degenerate lens.
- **Degenerate lens:** c⊥ = 0 (theory.md, Remark 1 and convention 4), read
  numerically as ‖c⊥‖ ≤ 1e-12·‖b‖ (`DEGENERATE_RTOL` in `lens/geometry.py`, as used for
  R6). A zero bias meets it exactly.
- **r_eff(d):** √(‖c⊥‖² + Hε)/‖A d‖ = r*(d)·√(1 + κ) (convention 6). For a degenerate
  lens this is the ε-limited width of convention 4.
- **Lens distance:** ρ_eff = ‖A(z − z*)‖²/(‖c⊥‖² + Hε), for every lens. Within one
  model ρ = (1 + κ)·ρ_eff (convention 2), a constant factor. The ordering by ρ_eff is
  therefore identical to the ordering by ρ wherever ρ is defined, and ρ_eff stays
  finite for a degenerate lens. Every set, rank and correlation below uses ρ_eff.
- **Jacobian error:** per test state, ‖J_s − J_std‖_F/‖J_std‖_F, on the full 12 × 16
  standardised Jacobian of the scaled-increment map.
- **R:** the median Jacobian error over the ceil(0.1 n) test states with the smallest
  lens distance, divided by the median over the ceil(0.5 n) with the largest.
- **rel-MSE:** MSE / mean per-coordinate variance of the standardised targets, on the
  validation and test sets.
- **Cells:** for each initialisation, the four (architecture, depth) combinations, of 5
  seeds each. A statement **holds in a cell** if it holds in at least 4 of the 5 seeds.
- **Evaluation point:** the early-stopping budget's model, unless stated otherwise.

## Predictions

Predictions 1–4 are quoted exactly from `docs/protocol_amendments_v3.md`, item 3. Each
is followed by its rule.

### Prediction 1

> Zero bias: the lens centre sits within 0.1 standardised units of the data mean
> (hover). Median Jacobian error over the 10% of test states nearest the lens (smallest
> rho) is at least 3x the median over the 50% farthest. Fixed rho cut-offs were dropped
> for the same high-dimension reason as in item 2.

**Rule.** A zero-bias model passes if both clauses hold:

- (a) ‖z* − μ‖ ≤ 0.1, the Euclidean norm in standardised coordinates;
- (b) R ≥ 3.

Prediction 1 holds in a zero-bias cell if both clauses hold in at least 4 of its 5
seeds. **Prediction 1 holds if it holds in every zero-bias cell.**

Reported alongside: each clause separately, per cell; ‖z* − hover_std‖ next to
‖z* − μ‖.

### Prediction 2

> PyTorch default: the initial r* lies within a factor of 2 of the sampling-check
> estimate, and the error ratio in (1) is smaller than under zero bias.

**Rule.**

- (a) For every PyTorch-default model, the initial median of r*(d) over 64 random unit
  directions (`numpy.random.default_rng(0)`; theory.md convention 4) lies in
  [0.935/2, 2 × 0.935]. 0.935 is the initial-lens check's median at H = 128, k = 16.
- (b) In every cell, R_torch < R_zero, paired by seed, in at least 4 of the 5 seeds.
  There is no accuracy matching; both arms' held-out rel-MSE is reported.
- Prediction 2 holds if (a) holds for every PyTorch-default model and (b) holds in
  every cell.

Clause (a) is known to hold by construction (see What has been seen) and carries no
evidential weight; the evidence is clause (b).

### Prediction 3

> Coverage and sharpness computed from the weights rank the 40 surrogates by
> affected-data fraction with Spearman rho >= 0.7.

**Rule.**

- **Score:** sharpness D(u_min)/r_eff(u_min).
- **Affected-data fraction:** the fraction of a model's test states whose Jacobian error
  exceeds 3× the median error of its 50% farthest states.
- **Test:** over all 40 surrogates, a positive Spearman correlation, ρ_s ≥ +0.7, between
  the score and the affected-data fraction. Ties get average ranks
  (`scipy.stats.spearmanr`). A negative correlation fails.
- Reported alongside: coverage (2/π)·arctan(D/r*) along u_min, both quantities along d₁,
  and the same Spearman correlation within each initialisation (n = 20 each), without a
  rule. The within-initialisation values are reported because a difference between the
  two initialisations alone could produce ρ_s ≥ 0.7 over all 40.

### Prediction 4

> The stack's first-block feature is attenuated, with E1b's median ratio of about 0.2
> as the point prediction.

**Scope:** the NormedLinear 3-block models. The quote names "the stack". In the
pre-norm residual model, the skip carries the un-normalised E z + b into block 1's
output, so its first-block feature is not that of a stack.

**The ratio** (primary; unit-free). Along z(t) = z* + t·u_min, with z* and u_min from
block 1's lens:

- s_out(t) = ‖∂y/∂t‖, with y in standardised units;
- s_1(t) = ‖∂h₁/∂t‖, with h₁ block 1's output in feature space (no head);
- P = s(0)/median of s(t), over 2,001 points uniform on [−D(u_min), D(u_min)], with s(0)
  evaluated exactly at t = 0;
- ratio = P_out/P_1.

**Rule.**

- In each of the two NormedLinear 3-block cells (both initialisations), the ratio is
  below 1 in at least 4 of the 5 seeds. Prediction 4 holds if both cells hold.
- **The point prediction:** the median ratio over the five zero-bias NormedLinear
  3-block models is reported as consistent with 0.2 if it lies in [0.1, 0.4].
- Reported without a rule:
  - the same ratio for the pre-norm 3-block models;
  - the head-on-block-1 ratio S_out/S_block1, where S = ‖∂y/∂t‖ at t = 0 and S_block1
    is taken for the model truncated after block 1, with the head applied to block 1's
    output.
- E1b's code and notes are not in the repository. Both ratios approximate E1b's
  measure.

### Prediction 5: the lens widens during training (zero bias)

Along u_min, r_eff(u_min)/D(u_min) rises by at least 10× from its initial value and
reaches at least 0.1, in at least 4 of 5 seeds per architecture, with κ ≤ 0.1 at the end.

**Rule.**

- **Per run:** at the end, the ratio is at least 10× its value at step 0 and at least
  0.1, and κ ≤ 0.1. A lens still degenerate at the end (see Definitions) has κ = ∞ and
  fails.
- **Primary test, the long-budget subset:** the end is step 500,000. An architecture
  holds if at least 4 of its 5 seeds pass, and prediction 5 holds if both architectures
  hold.
- **At early stopping (reported):** the same rule on the 20 zero-bias runs of the grid,
  with the end at the returned parameters' step, per zero-bias cell.
- **Secondary:** the same quantities along d₁ are reported.
- **Comparison only:** trained TD-MPC2 encoder lenses (R6, exploratory E3) sit at
  r_eff(d₁)/D of 0.25–1.17. That range is not a threshold.

**The race (reported, not tested).** If the lens widens enough, with r_eff(u_min)/D
toward the order of 1, prediction 1's R is expected to fall below 3. Two things are
reported, and the paper says which happened:

- for each zero-bias cell at early stopping, the seeds counted by (prediction 1
  passes, the early-stopping version of prediction 5 passes), with diverged seeds
  counted separately;
- for the long runs, R and r_eff(u_min)/D at every 10,000-step snapshot.

## Hover linearisation check and H1

**The matrices.** A := ∂y/∂x and B := ∂y/∂u, in physical units, taken by autodiff the
same way for the truth (the RK4 y-map) and for the surrogate. For each surrogate,
reported at hover:

- the relative Frobenius errors of A and of B;
- **sign agreement on a mask of physically meaningful entries:**
  - the mask is the set of entries of the continuous vector field's Jacobians ∂f/∂x and
    ∂f/∂u at hover (from the simulator's vector field, not the RK4 map) with
    |∂f| > 1e-3 × max|∂f|, per matrix;
  - on the mask, the signs of the surrogate's (A, B) are compared with those of the
    true (A, B);
  - the stage first checks that, on the mask, every true entry has the vector field's
    sign;
  - RK4 adds O(dt) cross-terms that are zero in the physics, for example
    ∂y_p/∂θ ≈ g·dt/2, so the mask keeps sign errors in the physics apart from these
    integration terms;
- the number of sign disagreements on the unmasked set (entries with |true y-map| >
  1e-3 × max|true y-map|, per matrix), reported without a rule. It includes the O(dt)
  integration terms;
- the continuous-time LQR gain from the surrogate's (A, B) against the true gain, with
  Bryson Q = diag(1/h_x²) and R = diag(1/(0.5 u0)²) from the half-widths above;
- the closed-loop eigenvalues of the surrogate's gain on the true (A, B): stable or not,
  and the spectral abscissa;
- **"no stabilising gain"**, recorded when the Riccati solve for the surrogate's (A, B)
  fails or its gain does not stabilise the surrogate's own (A, B).

**H1** (the claims ledger: "under zero bias, the surrogate's hover linearisation is
wrong in sign or stability").

- **Rule:** H1 holds if, in every zero-bias cell, at least 4 of the 5 seeds show at least
  one of:
  - a sign disagreement on the mask, in A or B;
  - no stabilising gain;
  - an unstable true closed loop under the surrogate's gain.
- Reported regardless: the zero-bias against PyTorch-default comparison of the (A, B)
  errors, paired by seed.

**Trims.** These are the 200 no-drag steady-flight trims committed in
`results/p1/trims/trims.csv` (SHA-256
5b5f71051dd4780fac6fb33b16a27154fdb4703a66c2acdd5d334f73305c9f95). They were sampled
before any training from `numpy.random.default_rng(20261005)`.

- Position, velocity and yaw are uniform in the training box. Roll, pitch and rates are
  0, and thrust is u0.
- Every trim lies inside the box and is a trim of the RK4 map.
- The true linearisation depends on a trim only through yaw.
- **Reported per trim:**
  - the errors of A and B;
  - sign agreement on that trim's vector-field mask;
  - the trim's lens distance ρ_eff;
  - the variation error ‖(J_s(trim) − J_s(hover)) − (J_t(trim) − J_t(hover))‖_F/‖J_t(hover)‖_F.
- **Reported per model:** the medians, and Spearman correlations with lens distance.
- No rule is attached to the trims.

## Gate G2 (18 October 2026)

- G2 passes if prediction 1 holds in at least 2 of the 4 zero-bias cells. P-III then
  runs on the cells where it holds.
- G2 is a planning gate, not a scientific claim. It decides where P-III's effort goes.
  Prediction 1's result is reported per cell either way.

## Stopping rules

- **Corollary 1.** The analysis stops on any Corollary 1 deviation above 1e-10 on a
  trained model whose lens is not degenerate (see Definitions).
  - The deviation is relative: on each line, the maximum over its grid of
    ‖ĥ(s) − ĥ_Cor1(s)‖₂/‖ĥ_Cor1(s)‖₂, with ĥ the LayerNorm output computed from the
    weights and ĥ_Cor1 Corollary 1's closed form with the line's own s*, c⊥,ℓ, κ_ℓ and
    r*_ℓ. The maximum is then taken over the lines.
  - The lines run through μ and through 10 training states, along d₁.
  - Such a deviation indicates a code error. On random first layers, the check stayed at
    or below 4e-12 for κ from 1e-4 to 1e10 and for ‖c⊥‖ just above the degeneracy
    tolerance (`results/p1/corollary1_stress/`).
- **Hover mask.** The hover stage stops if, on the vector-field mask at hover or at any
  trim, a true y-map entry's sign differs from the vector field's. Such a difference
  indicates a code error.
- **Manifests.** Every stage stops on any data-manifest mismatch.
- **Merging runs.** Merging the per-run outputs stops on any missing run, any SHA-256
  mismatch, more than one commit, or a dirty working tree.
- **Divergence.** A run whose training or held-out loss, or any of whose parameters,
  becomes non-finite at an evaluation is stopped and recorded as diverged.
  - It fails every rule it enters: a per-seed rule does not count the seed, a pooled
    rule (prediction 3) fails, and H1 does not count it.
  - It is reported, and no seed is replaced.
- **Interruptions.** An interrupted run (crash, reboot, power loss) is restarted from
  step 0 with the same seed. Resuming from a snapshot is not used.
- **Atomic outputs.** Every completed run writes its outputs atomically, and its
  per-run SHA-256 manifest last. A driver skips completed runs whose manifests verify.

## Reported regardless of outcome

For every model:
- the training history and lens log;
- held-out rel-MSE;
- the final lens: z*, κ, widths, and coverage and sharpness along u_min and d₁;
- R with both sets' statistics, the affected-data fraction, and the Jacobian error
  distribution (quantiles);
- the Corollary 1 deviations;
- both prediction-4 ratios for the 3-block models;
- the hover check, H1's per-model outcome and the trim check;
- prediction 5's per-run records, for both training stages;
- divergences.

For every stage: the config, the git commit, the package versions, the data manifest and
the per-run SHA-256 manifests.

## What has been seen before this pre-registration

No quadrotor training data has been generated, and no surrogate has been trained on
quadrotor data. The following has been seen:

- **Initial lenses** (`results/initial_lens/`, `results/p1/init_lens/`), from
  initialisation alone with synthetic i.i.d. uniform inputs.
  - Along u_min, r_eff/D at initialisation:
    - zero bias: median 0.0087 (p5–p95 0.0083–0.0091; seeds 0–4: 0.0082–0.0089),
      degenerate;
    - PyTorch default: 0.37 (0.34–0.40).
  - The PyTorch-default initial median r* is 0.94 (0.86–1.01); seeds 0–4 give
    0.92–0.97. Prediction 2's clause (a) is therefore known to hold for every
    PyTorch-default model.
- **Runtime benchmarks, throughput and determinism checks** on random targets and on a
  synthetic linear plant: timings and byte comparisons only (`results/p1/benchmark/`,
  `results/p1/benchmark_bs2048/`, `results/p1/throughput/`,
  `results/p1/determinism/`).
- **The Corollary 1 stress test** on random first layers with synthetic inputs
  (`results/p1/corollary1_stress/`): maximum relative deviation 4.0e-12.
- **R6** (`results/r6/criterion/`): G1 did not pass (0 of 4). No TD-MPC2 checkpoint met
  Inside, Populated and Sharp; Sharp failed for all 15, with D/r_eff(d₁) 0.85–4.08.
- **R6 exploratory E3** (`results/r6/exploratory/`): trained TD-MPC2 encoder lenses at
  r_eff(d₁)/D 0.25–1.17, against 1e-4–0.085 at their initialisation.
- **The simulator:**
  - its hover linearisation, vector-field Jacobians and Jacobians, in the plant tests;
  - the 200 trims, checked on the simulator only.
- **Synthetic tests and the pre-flight:** surrogates trained on a synthetic linear plant
  with the quadrotor's dimensions, in the pipeline tests. Also the complete pipeline
  (`experiments/p1_quadrotor/preflight.py`; all 40 + 10 models with cut budgets),
  checked for completeness, not for its outcomes, which say nothing about the
  quadrotor.
