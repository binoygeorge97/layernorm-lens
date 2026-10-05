# DRAFT v3 (final): Pre-registration P-I, quadrotor surrogates

**Status: draft for the author and advisor. This is not a pre-registration file.**
When agreed, it becomes `prereg/p1.md`, committed alone. The author tags it
`prereg-p1`, and only then may any P-I stage touch quadrotor data.

- v1 (4 Oct 2026, def116d): reconstructed predictions 1–4 from `docs/plan.md`.
- v2 (4 Oct 2026): predictions 1–4 quoted exactly from the author's protocol amendments
  v3 (`docs/protocol_amendments_v3.md`, commit d6b822b, item 3), which win over plan.md
  where they differ. Also:
  - prediction 5 rewritten (u_min, the long-budget subset, a race with prediction 1);
  - drag off, minibatch Adam, the evaluation interval and the budgets stated;
  - every open decision given a recommendation.
- v3 (5 Oct 2026): the author's decisions on D1–D18 applied (accepted as recommended:
  D1, D2, D4–D8, D11–D14, D17, D18; changed: D3, D9, D10, D15, D16). The trim code was
  written and the trims were sampled before any training. `prereg/p1.md` is written
  from this version.

**Provenance of the predictions.** `docs/protocol_amendments_v3.md` was committed on
4 October 2026 (d6b822b), after G1 was evaluated on 3 October. Its R6 item (item 2) had
been tagged in `prereg/r6.md` (prereg-r6) before any R6 data. Its P-I items precede any
quadrotor data: none had been generated when this was written.

Everything below the quotes (how each prediction is evaluated) was settled by the
author's decisions of 5 October (end). The amendments say the thresholds are confirmed "after the
initial-lens sampling check, never after seeing trained quadrotor models". The check is
done (`results/initial_lens/`), and no quadrotor model has been trained.

## Question

Gate G2 (18 Oct): do zero-bias quadrotor surrogates concentrate Jacobian error at
hover, near the first-layer lens?

And the race behind it: training both fits the data and can repair the lens. Does the
early-stopping budget freeze the zero-bias defect in (prediction 1), or does training
widen the lens until it no longer matters (prediction 5)?

## Plant and data

- **Plant:** `plants/quadrotor.py` at the tagged commit.
  - 12 states: position, velocity, ZYX Euler angles, body rates.
  - 4 inputs: rotor thrusts in [0, f_max].
  - The author-approved Crazyflie 2.x parameters from gym-pybullet-drones' `cf2x.urdf`
    (889ce4a5c068ae4d811df1442ceb4f4d6cdf43eb), with g = 9.81.
  - RK4 at dt = 0.01 s with zero-order hold.
- **No aerodynamic drag (a modelling choice).**
  - No linear drag coefficient is in the approved parameter set.
  - The surrogates learn whatever map the simulator defines, so the lens question does
    not depend on drag.
  - For scale: the URDF's rotor drag, linearised at hover, would give about
    5.6e-3 N/(m/s) in x and y, so 2% of the weight (0.265 N) at the box's 1 m/s.
  - One consequence: without drag, level flight at any constant velocity, position and
    yaw is a trim with hover thrust and zero attitude (used by the hover check's trims,
    D16).
- **Inputs to the surrogate:** z = (x, u) ∈ R¹⁶ (k = 16), z-scored per coordinate with
  the training split's mean and standard deviation.
  - The dynamics are translation-invariant, so the position columns of the true
    Jacobian are zero; the surrogate must learn that too.
- **Target** (amendments item 1): y = (x_{t+1} − x_t)/dt ∈ R¹², z-scored the same way.
  Main-text Jacobian errors are for this map; discrete-map values go to the appendix.
- **Ground truth:** J = ∂y/∂z by forward-mode autodiff of the simulator, in standardised
  coordinates J_std = diag(1/σ_y) J diag(σ_z).
- **Sampling** (D3):
  - i.i.d. uniform in a box around hover: position ±1 m, velocity ±1 m/s, roll and
    pitch ±30°, yaw ±180°, body rates ±2 rad/s;
  - thrust u_i = u0 (1 ± 0.5) with u0 = m g/4, the hover thrust: [0.5 u0, 1.5 u0],
    symmetric about hover thrust and inside [0, f_max] = [0, 2.25 u0], so never clipped
    (D3);
  - every range is symmetric about hover (position 0, velocity 0, attitude 0, yaw 0,
    rates 0, thrust u0), so the standardised mean is hover in every input dimension, up
    to sampling noise;
  - 20,000 train, 5,000 validation and 5,000 test states, from
    `numpy.random.default_rng(0)` drawn in that order.
  - Every coordinate is independent and uniform, so the standardised inputs are i.i.d.
    U(−√3, √3) up to sampling noise. Hover is the box centre, so the data mean is
    hover, which is where a zero-bias lens starts (theory.md, Remark 1).
- **Storage:** the data stays outside the repository (`data/p1/`; on TACC,
  `$SCRATCH`). Its SHA-256 manifest (`results/p1/data_manifest.csv`) is committed, and
  every stage verifies it before use.

## Surrogates

- **Architectures:**
  - (a) Pre-norm residual block: h = E z + b; then per block,
    h ← h + W2 GELU(W1 (γ LN(h) + β) + b1) + b2; finally y = Wo h + bo.
  - (b) TD-MPC2-style NormedLinear stack: Linear → LayerNorm (affine) → Mish, with a
    Linear head.
- **Initialisations:**
  - `zero_bias`: PyTorch-default weights U(±1/√fan_in), every bias 0.
  - `torch_default`: weights and biases U(±1/√fan_in).
  - The two arms of a seed share every weight, and the minibatch stream too, so they
    are paired.
  - The first layer (E, b) is the first draw for both architectures and both depths.
    The initial lens of a seed is therefore the same in all four cells.
- **Grid** (amendments, "P-I amended"): H = 128; 1 block or a 3-block stack; ε = 1e-5;
  5 seeds (0–4). That is 2 × 2 × 2 × 5 = 40 surrogates.
- **The lens** is the first LayerNorm's (theory.md, Theorem 1). Lens distance is ρ, or
  ρ_eff = ‖A(z − z*)‖²/(‖c⊥‖² + Hε) for a degenerate lens.
- **Directions:**
  - u_min is the lens's narrowest principal direction: the right singular vector of A
    for its largest singular value. It is recomputed from the current weights wherever
    it is used.
  - d₁ is the training inputs' principal direction (r6.md). It is poorly determined
    here, because the standardised inputs' covariance is close to I.
  - D(d) = (q97.5 − q2.5)/2 of the training inputs' projections on d.

## Training

- **Optimiser:** minibatch Adam (the author's decision of 4 Oct 2026), float64; batch
  2,048; lr 3e-3; (β1, β2, ε) = (0.9, 0.999, 1e-8).
  - Each step draws 2,048 distinct training rows afresh, from the run's seed and the
    step number. There are no epochs.
- **Evaluation interval:** every 500 steps, the held-out one-step MSE on the full
  validation set (5,000 states).
- **Early stopping** (amendments item 6): patience 10% of the budget, so 10,000 steps;
  tolerance 1% on the held-out one-step MSE.
  - An evaluation counts as an improvement only if it is below 99% of the MSE at the
    last improvement.
  - The returned parameters are those with the lowest held-out MSE.
- **Budgets:**
  - The 40 runs: early stopping within at most 100,000 steps (plan.md's converged
    budget). This is the early-stopping budget at which predictions 1–4 are evaluated.
  - **The long-budget subset:** zero bias, 1 block, both architectures, seeds 0–4: 10
    runs. Each trains for a fixed 500,000 steps (5× the 40 runs' maximum) with early
    stopping off, logging the lens throughout. Parameter snapshots are saved every
    10,000 steps.
  - Cost on the laptop CPU, from the minibatch benchmark on random targets
    (`results/p1/benchmark_bs2048/`; upper bounds with no early stop, lens logging
    included): at most 30.2 h for the 40 runs (0.016–0.042 s per step) and at most
    24.5 h for the long-budget subset (about 2.5 h per run). That is about 55 h in all,
    so the laptop can finish before G2. TACC (`slurm/`) is the fallback.
- **Lens log**, every 500 steps (each evaluation):
  - z*, κ, ‖c⊥‖ and the principal widths;
  - along u_min and along d₁: r*, r_eff, D, r_eff/D, and the susceptibility
    S = ‖J(z*) d‖.

## Initial values (from initialisation alone)

Computed before any quadrotor data exists (`run.py init_lens`,
`results/p1/init_lens/`). The table covers seeds 0–999 of each initialisation, with the
initial first layer at H = 128, k = 16, ε = 1e-5. The inputs are 20,000 i.i.d.
U(−√3, √3)¹⁶ draws, z-scored, standing in for the training inputs (see Sampling).

| Initialisation | r_eff(u_min)/D(u_min): p5 / p50 / p95 | seeds 0–4 | r_eff(d₁)/D(d₁), p50 | κ | ‖z*‖, p50 | median r*(d), p5 / p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| zero_bias | 0.0083 / 0.0087 / 0.0091 | 0.0085, 0.0089, 0.0085, 0.0082, 0.0083 | 0.0111 | ∞ (degenerate) | 0 | 0.022 / 0.022 / 0.023 (ε-limited, √(Hε)/‖A d‖) |
| torch_default | 0.34 / 0.37 / 0.40 | 0.380, 0.380, 0.366, 0.363, 0.356 | 0.47 | 5.5e-4 | 0.37 | 0.86 / 0.94 / 1.01 |

- D(u_min) ≈ 1.95 and D(d₁) = 1.99. The inputs' top-two covariance eigenvalue ratio is
  1.015, so d₁ is essentially arbitrary.
- The zero-bias row along d₁ matches v1's random-direction value (median 0.0113).
- The PyTorch-default median r* agrees with the initial-lens check (median 0.935 over
  its own 1,000 draws).

## Predictions

Predictions 1–4 are quoted exactly from the protocol amendments v3, item 3. Below each
quote is how it is evaluated (the decisions are referenced).

- **Cells:** each initialisation has 4 cells (architecture × depth) of 5 seeds.
- **"Holds in a cell"** means it holds in at least 4 of the 5 seeds (D14).
- **Evaluation point:** predictions 1–4 are evaluated at the early-stopping budget, on
  the returned parameters. The one exception is clause 2(a), which is about the
  initialisation.

### Prediction 1 (G2's core)

> 1. Zero bias: the lens centre sits within 0.1 standardised units of the data
>    mean (hover). Median Jacobian error over the 10% of test states nearest
>    the lens (smallest rho) is at least 3x the median over the 50% farthest.
>    Fixed rho cut-offs were dropped for the same high-dimension reason as in
>    item 2.

Evaluated on each zero-bias model (D7):

- **(a) centre:** ‖z* − μ‖ ≤ 0.1, the Euclidean norm in standardised input
  coordinates, with μ the training inputs' mean.
- **(b) concentration:** R ≥ 3. R is the median relative Frobenius Jacobian error over
  the ceil(0.1 n) test states with the smallest lens distance, divided by the median
  over the ceil(0.5 n) with the largest.
  - The lens distance is ρ, or ρ_eff if the lens is still degenerate.
  - The error is ‖J_s − J‖_F/‖J‖_F per state, for the full 12 × 16 Jacobian (D5).
- A seed counts if both (a) and (b) hold. Each clause is also reported on its own.

### Prediction 2

> 2. PyTorch default: the initial r* lies within a factor of 2 of the
>    sampling-check estimate, and the error ratio in (1) is smaller than under
>    zero bias.

Evaluated (D8):

- **(a)** For each PyTorch-default model, the initial median of r*(d) over 64 random
  unit directions (theory.md convention 4) lies in [0.47, 1.87]. That is a factor of 2
  around the sampling check's median, 0.935, at H = 128, k = 16
  (`results/initial_lens/summary.csv`; the theory estimate is 0.931).
  - Clause (a) depends only on the initialisation, so its outcome is already known:
    seeds 0–4 give 0.97, 0.95, 0.95, 0.95 and 0.92. All 20 PyTorch-default models
    share these five first layers, so (a) holds for every one of them.
  - It confirms that the surrogates start where the sampling check said they would.
- **(b)** In each of the 4 cells, R_torch < R_zero in at least 4 of the 5 seeds,
  paired by seed (shared weights and minibatches).
  - Each arm is taken at its own early-stopping budget. There is no accuracy matching;
    both arms' held-out rel-MSE is reported.

### Prediction 3

> 3. Coverage and sharpness computed from the weights rank the 40 surrogates
>    by affected-data fraction with Spearman rho >= 0.7.

Evaluated once, over all 40 surrogates (D9):

- **Score:** sharpness along u_min, D(u_min)/r_eff(u_min). Along one direction,
  coverage (2/π)·arctan(D/r*) is a monotone function of it (r* and r_eff differ by
  √(1 + κ)), so the two give the same ranking. Both are reported, as are the d₁
  versions.
- **Affected-data fraction:** the fraction of a model's test states whose Jacobian error
  exceeds 3× the median error of its 50% farthest states (prediction 1's far set and
  factor).
- **Test:** a positive Spearman correlation, ρ_s ≥ +0.7, between the score and the
  affected fraction over the 40 surrogates. Ties get average ranks
  (`scipy.stats.spearmanr`). A negative correlation fails, whatever its size.
- Why a positive relation is expected: along a Lorentzian of width r, the slope is at
  least 3× its value at the data edge D wherever |s| ≤ √((D² − 2r²)/3). That band grows
  from nothing at r = D/√2 to D/√3 as r → 0, so a sharper lens affects more of the
  data, up to a saturation.

### Prediction 4

> 4. The stack's first-block feature is attenuated, with E1b's median ratio
>    of about 0.2 as the point prediction.

Evaluated on the 3-block models (D10):

- **The ratio** (D10): S_out/S_block1 along the line z* + t·u_min through block 1's
  lens centre, with S the peak slope at the lens centre, ‖∂y/∂t‖ at t = 0.
  - S_out is taken for the full model.
  - S_block1 is taken for the model truncated after block 1, with the head applied to
    block 1's output.
  - A ratio below 1 means the later blocks attenuate the slope that the first block's
    lens puts at its centre.
- **Attenuated:** in each of the four 3-block cells (both architectures, both
  initialisations), the ratio is below 1 in at least 4 of the 5 seeds.
- **The point prediction:** the median ratio over the ten zero-bias 3-block models is
  reported against 0.2, as consistent if it lies within a factor of 2 ([0.1, 0.4]).
- E1b's code and notes are not in the repository or its history: E1b is named only in
  theory.md's Scope remark. This ratio approximates E1b's measure. The sharpness ratio
  (peak/median of the speed profiles) is reported alongside.

### Prediction 5 (new): the lens widens during training

Along u_min, the lens's narrowest principal direction, r_eff(u_min)/D(u_min) rises by
at least 10× from its initial value and reaches at least 0.1, in at least 4 of 5
seeds per architecture, with κ ≤ 0.1 at the end.

- **Per run:** at the end, the ratio is at least 10× its value at step 0 and at least
  0.1, and κ ≤ 0.1. A lens still degenerate at the end has κ = ∞ and fails.
- **Primary test (the long-budget subset):** zero bias, 1 block. The end is the last
  step of the 500,000-step budget. P5 holds for an architecture if at least 4 of its 5
  seeds pass, and P5 holds if it holds for both architectures. Each architecture is
  also reported on its own.
- **At the early-stopping budget (the 40 runs' trajectories):** on every zero-bias run,
  with the end at the returned parameters' step. Reported per zero-bias cell, at least
  4 of 5 seeds. This is the race's other half, read with prediction 1 (below).
- **Thresholds, from the initial values above:**
  - The zero-bias start along u_min is 0.0087 (median over 1,000 seeds; 0.0082–0.0089
    for seeds 0–4), so 10× is about 0.087, and the 0.1 floor is the binding condition.
  - κ ≤ 0.1 means ‖c⊥‖² ≥ 10 Hε: the bias offset, not ε, sets the width.
- **Secondary and comparison:**
  - The same quantities along d₁ are secondary and reported.
  - Trained TD-MPC2 encoder lenses (R6, exploratory E3) sit at r_eff(d₁)/D of
    0.25–1.17. That range is a comparison, not a threshold.
- **Scope:** prediction 5 is made for zero bias only. The PyTorch-default start is
  already at 0.37 along u_min (0.36–0.38 for seeds 0–4), a factor of 2.7 short of 1.
  Its trajectory is reported (D11).
- **Reported regardless,** for every run of both stages: the trajectories of κ, ‖z*‖,
  ‖z* − μ‖, r_eff/D along u_min and d₁, and S. For the long runs, also prediction 1's
  statistic R and ‖z* − μ‖ at every 10,000-step snapshot (`analyse_long`).

### The race between fitting and repair (predictions 1 and 5)

Prediction 1 is evaluated at the early-stopping budget, and prediction 5 over training.
They are linked.

- **Expected relation:** if the lens widens enough, with r_eff(u_min)/D(u_min) toward
  the order of 1, prediction 1's error ratio R is expected to fall below 3. A lens as
  wide as the data no longer singles out the states near its centre.
- If R ≥ 3 at early stopping, fitting has won the race up to the budget at which
  surrogates are normally used: the defect is still there when training stops.
- The long-budget runs show whether repair wins later. The race is reported, not
  tested (D13):
  - each zero-bias cell's 2 × 2 table at early stopping, prediction 1 (holds or not)
    against prediction 5's early-stopping version (passes or not);
  - for the long runs, R and r_eff(u_min)/D(u_min) against steps.

The paper will report which happened.

### Pipeline check: Corollary 1 on trained models (plan.md, "Measure")

On every trained model whose lens is non-degenerate, the first LayerNorm's output
matches Corollary 1 to 1e-10 relative, along lines through μ and through 10 training
states along d₁. This is the R6 D7 (4) rule. A failure indicates a code error, not a
result, and the analysis stage stops (D18).

## Hover linearisation check (amendments: new)

For each surrogate, its (A, B) at hover is compared with the simulator's. Both are taken
by autodiff, on the y-map, in physical units (D16). The comparison reports:

- the relative Frobenius error of A and of B;
- sign agreement on every entry whose true magnitude exceeds 1e-3 × max|truth| (per
  matrix);
- the LQR gain from the surrogate's (A, B) against the true gain. This is
  continuous-time, with Bryson Q and R from the sampling half-widths: Q = diag(1/h_x²),
  R = diag(1/(0.5 u0)²);
- the closed-loop eigenvalues of the surrogate's gain on the true linearisation: stable
  or not, and the spectral abscissa;
- **"no stabilising gain"**, recorded when the surrogate's (A, B) has no stabilising LQR
  solution: the Riccati solve fails, or its gain does not stabilise the surrogate's own
  (A, B). It counts as a failure for H1.

**Trims** (D16): 200 no-drag steady-flight trims, sampled before any training from
`numpy.random.default_rng(20261005)` and committed in `results/p1/trims/trims.csv`
(SHA-256 5b5f71051dd4780fac6fb33b16a27154fdb4703a66c2acdd5d334f73305c9f95, commit
0966917).

- Position, velocity and yaw are uniform in the training box. Roll, pitch and rates are
  0, and thrust is u0.
- Every trim lies inside the training box, and is a trim of the RK4 map to 1.1e-16.
- **What is reported per trim:** the errors of A and B, sign agreement, the trim's lens
  distance, and the variation error.
- **Variation error:** the true linearisation depends on the trim only through yaw. The
  error in the surrogate's variation across trims is therefore reported as
  ‖(J_s(trim) − J_s(hover)) − (J_t(trim) − J_t(hover))‖_F/‖J_t(hover)‖_F.
- **Per model:** its median, and its Spearman correlation with the trims' lens distance.

**H1, the ledger's new claim:** "under zero bias, the surrogate's hover linearisation is
wrong in sign or stability." Pre-registered (D16). H1 holds if, in each zero-bias
cell, at least 4 of the 5 seeds have at least one of:
  - a sign disagreement in A or B at hover (above the threshold);
  - no stabilising gain;
  - an unstable true closed loop under the surrogate's gain.
- Reported regardless: the paired comparison of the (A, B) errors, zero bias against
  PyTorch default, by seed.

## Gate G2

G2 passes if prediction 1 holds in at least 2 of the 4 zero-bias cells. P-III then runs on
the cells where it holds.

G2 is a planning gate, not a scientific claim. It decides where P-III's effort goes;
prediction 1's result is reported per cell regardless.

## Reported regardless of outcome

For every model:
- the training history and lens log (both stages);
- the final lens: z*, κ, widths, and coverage and sharpness along u_min and d₁;
- R with both sets' statistics, the affected-data fraction, and the Jacobian error
  distribution;
- the Corollary 1 deviations and attenuation;
- the full hover check;
- the prediction 5 run records (`p5_*.csv`);
- for every stage: the config, the commit, the package versions, the data manifest,
  and the per-run output manifests.

## Decisions (settled by the author on 5 October 2026)

- Accepted as recommended in v2: D1, D2, D4, D5, D6, D7, D8, D11, D12, D13, D14, D17,
  D18.
- **D3:** accepted. The thrust range is symmetric about hover thrust, and every box range
  is listed (Sampling).
- **D9:** accepted, with the direction stated: a positive Spearman ρ_s ≥ 0.7, with ties
  given average ranks.
- **D10:** E1b's definition was searched for and not found. The ratio is S_out/S_block1
  at block 1's lens centre along u_min, and 0.2 is consistent if within a factor of 2.
- **D15:** G2 at 2 of the 4 zero-bias cells, with P-III on those cells. G2 is a planning
  gate.
- **D16:** accepted, with three additions:
  - "no stabilising gain" counts as a failure for H1;
  - the trims lie inside the training box;
  - the variation across trims is reported against lens distance.

  The trims were sampled before any training.
- **Compute:** the laptop; TACC is not needed. The scripts are kept.
- Aggregations made explicit in code (`p1_rules.py`):
  - prediction 2 holds if (a) holds for every model and (b) holds in every cell;
  - prediction 4 holds if every 3-block cell holds;
  - H1 holds if every zero-bias cell holds;
  - prediction 5 holds if both architectures hold.
