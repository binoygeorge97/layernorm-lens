# DRAFT v2: Pre-registration P-I, quadrotor surrogates

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

Everything below the quotes (how each prediction is evaluated) is a proposal, listed as
a decision at the end. The amendments say the thresholds are confirmed "after the
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
  - thrust u_i = u0 (1 ± 0.5) with u0 = m g/4 (within [0, f_max] = [0, 2.25 u0], so
    never clipped);
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
quote is how it is evaluated (proposed; the decisions are referenced).

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
- **Test:** Spearman's ρ_s between the score and the affected fraction is at least 0.7.
- Why a positive relation is expected: along a Lorentzian of width r, the slope is at
  least 3× its value at the data edge D wherever |s| ≤ √((D² − 2r²)/3). That band grows
  from nothing at r = D/√2 to D/√3 as r → 0, so a sharper lens affects more of the
  data, up to a saturation.

### Prediction 4

> 4. The stack's first-block feature is attenuated, with E1b's median ratio
>    of about 0.2 as the point prediction.

Evaluated on the 3-block models (D10):

- **The ratio:** along z* + t·u_min, on a grid uniform in t over the data's extent, the
  sharpness (peak/median) of the output's speed ‖∂y/∂t‖, divided by that of block 1's
  normalised feature ‖∂ĥ₁/∂t‖. Block 1's profile is Corollary 1's Lorentzian.
- **Attenuated:** in each of the four 3-block cells (both architectures, both
  initialisations), the ratio is below 1 in at least 4 of the 5 seeds.
- **The point prediction:** the median ratio over the ten zero-bias 3-block models is
  reported against 0.2, as consistent if it lies within a factor of 2 ([0.1, 0.4]).
- E1b's exact definition of its ratio is not in the repository (question for the
  author). If it differs, it replaces this one.

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
    0.25–1.2. That range is a comparison, not a threshold.
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
by autodiff, on the y-map, in physical units (D16):

- the relative Frobenius error of A and of B;
- sign agreement on every entry whose true magnitude exceeds 1e-3 × max|truth| (per
  matrix);
- the LQR gain from the surrogate's (A, B) against the true gain. This is
  continuous-time, with Bryson Q and R from the sampling half-widths: Q = diag(1/h_x²),
  R = diag(1/(0.5 u0)²);
- the closed-loop eigenvalues of the surrogate's gain on the true linearisation: stable
  or not, and the spectral abscissa.

**Trims** (proposed, D16): 200 steady-flight trims, sampled once with a fixed seed from
the no-drag trim family inside the box. Positions, velocities and yaw are uniform;
attitude is level, rates are zero and thrust is u0. Each trim's errors are reported
against its lens distance. The trims are fixed before training, so they cannot be
chosen after seeing a lens.

**H1, the ledger's new claim:** "under zero bias, the surrogate's hover linearisation is
wrong in sign or stability." Proposed as a pre-registered prediction (D16):

- in each zero-bias cell, at least 4 of the 5 seeds have either:
  - at least one sign disagreement in A or B at hover (above the threshold), or
  - an unstable true closed loop under the surrogate's gain.
- Reported regardless: the paired comparison of the (A, B) errors, zero bias against
  PyTorch default, by seed.

## Gate G2

Proposed (D15): G2 passes if prediction 1 holds in at least 3 of the 4 zero-bias cells.

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

## Decisions for the author, each with a recommendation

1. **D1. Predictions 1–4's wording.**
   - Recommendation: the exact quotes above, with the operationalisations in D7–D10.
   - Why: the amendments are the author's text, and they win over plan.md.
   - Alternatives: none. v1's reconstructions are withdrawn; v1's prediction 3 (hover
     damage) became H1, and v1's prediction 4 (Corollary 1) became a pipeline check.
2. **D2. Plant.**
   - Recommendation: g = 9.81, no drag, RK4 at dt = 0.01 s, position among the inputs
     (k = 16).
   - Why: these are the approved parameters, the author's no-drag decision, and
     plan.md's k = 16 = 12 + 4.
   - Alternatives:
     - g = 9.8 (gym-pybullet-drones' value; not the approved one);
     - dropping position, since the dynamics do not depend on it. That would make
       k = 13 and break the plan's k = 16 and the initial-lens check's (H, k).
3. **D3. Sampling design.**
   - Recommendation: the i.i.d. uniform box above; 20,000 / 5,000 / 5,000; data seed 0.
   - Why:
     - The standardised inputs come out isotropic, so where the lens sits is not
       confounded with the data's shape.
     - The data mean is hover, which is what prediction 1's "(hover)" assumes.
   - Alternatives:
     - closed-loop trajectories (realistic, but anisotropic and correlated, and "near
       hover" then depends on the controller);
     - a Gaussian box (unbounded tails);
     - wider tilt (±45°).
4. **D4. Standardisation.**
   - Recommendation: a per-coordinate z-score from the training split.
   - Why: amendments item 1 says "standardised coordinates", and the initial-lens
     check's conventions assume unit-scale inputs.
   - Alternatives: fixed physical scaling by the box half-widths. For a uniform box it
     agrees with the z-score up to a factor of √3 and sampling noise, but it does not
     give unit variance.
5. **D5. Jacobian error.**
   - Recommendation: the relative Frobenius error ‖J_s − J‖_F/‖J‖_F per state, for the
     full 12 × 16 standardised Jacobian of the scaled-increment map.
   - Why: amendments item 1 (units), and the error is scale-free across states.
   - Alternatives:
     - the absolute Frobenius error (dominated by high-gain states);
     - A and B blocks separately (reported in the hover check);
     - physical units (appendix).
6. **D6. Direction.**
   - Recommendation: u_min is primary for predictions 3, 4 and 5; d₁ is secondary and
     reported. D is (q97.5 − q2.5)/2 of the training inputs' projections.
   - Why:
     - d₁ is undetermined for isotropic standardised inputs (top-two eigenvalue ratio
       near 1).
     - u_min is defined by the lens itself, and it is the direction of the largest
       amplification.
   - Alternatives: d₁ (r6.md's definition), or the median over the principal
     directions.
7. **D7. Prediction 1.**
   - Recommendation:
     - (a) ‖z* − μ‖ ≤ 0.1, Euclidean in standardised coordinates, μ the training mean;
     - (b) R ≥ 3 with ceil-sized 10% and 50% sets and medians;
     - both clauses per seed, at the returned parameters.
   - Why: this is the quote's literal reading. The Euclidean and Mahalanobis norms agree
     here up to noise, because the covariance is close to I.
   - Alternatives:
     - clause (b) alone for G2, with (a) reported (if migration should not fail G2);
     - distance to hover's standardised coordinates instead of μ.
8. **D8. Prediction 2.**
   - Recommendation:
     - (a) the initial median r* over 64 directions in [0.47, 1.87];
     - (b) R_torch < R_zero, paired by seed, in at least 4 of 5 seeds per cell, without
       accuracy matching.
   - Why:
     - 64 directions is the sampling check's number.
     - Pairing removes the weight and minibatch noise.
     - The amendments put matched-accuracy comparisons in P-II, not P-I.
   - Alternatives:
     - matching held-out rel-MSE within 20% (P-II's rule);
     - comparing the cells' median R.
9. **D9. Prediction 3.**
   - Recommendation:
     - score D/r_eff along u_min;
     - affected = error above 3× the model's far-set median;
     - one Spearman test over all 40, at least 0.7.
   - Why: it uses prediction 1's own sets and factor, so "affected" means "lens-like
     error concentration", independent of each model's overall accuracy.
   - Alternatives:
     - an absolute error threshold pooled over models (e.g. the pooled 90th
       percentile), which mixes in overall accuracy;
     - a score from the weights alone (the minimum principal width), which ignores D;
     - separate tests per initialisation (n = 20 each; low power).
10. **D10. Prediction 4.**
    - Recommendation:
      - the output-over-block-1 sharpness ratio along u_min;
      - attenuated if below 1 in at least 4 of 5 seeds in each of the four 3-block
        cells;
      - 0.2 reported as consistent within [0.1, 0.4] on the zero-bias median.
    - Why:
      - u_min is the closest analogue of a one-input lens.
      - The factor of 2 matches prediction 2's tolerance for an estimate.
    - Alternatives:
      - zero-bias cells only;
      - d₁;
      - block 3's ĥ instead of the output;
      - treating 0.2 as a pass/fail threshold.
    - **Please supply E1b's definition of its ratio.**
11. **D11. Prediction 5.**
    - Recommendation: as written above (fold 10, floor 0.1, κ ≤ 0.1), evaluated at the
      end.
      - Primary: the long-budget subset, per architecture.
      - At early stopping: per zero-bias cell, reported with prediction 1.
      - Zero bias only.
    - Why:
      - The floor sits an order of magnitude above the zero-bias start, and the fold
        guards against a start that is already wide.
      - Evaluating at the end, not at the best moment, avoids rewarding a transient.
    - Alternatives:
      - a floor of 0.25, the low end of TD-MPC2's range (the author made that range a
        comparison, not a threshold);
      - "reaches at any time" (reported as `ratio_max`);
      - κ ≤ 0.01;
      - also predicting the PyTorch-default arm.
12. **D12. The long budget.**
    - Recommendation: 500,000 steps, 5× the 40 runs' maximum, with snapshots every
      10,000 steps.
    - Why: every run is then trained to at least 5× its early-stopping budget. The
      benchmark puts the subset at no more than 24.5 h on the laptop, so about 55 h with
      the grid, which fits before G2.
    - Alternatives:
      - 300,000 steps (3×; cheaper, but less room past a late stop);
      - 1,000,000 steps (10×; doubles the cost).
13. **D13. The race.**
    - Recommendation: state the expected relation and report it (the 2 × 2 tables and
      the long-run traces), with no pass/fail test.
    - Why: it is a mechanism read across two predictions that are each tested.
    - Alternatives: a formal test, such as Spearman(R, r_eff/D) ≤ −0.5 across the 20
      zero-bias runs at early stopping.
14. **D14. The seeds rule.**
    - Recommendation: "holds in a cell" means at least 4 of 5 seeds.
    - Why: it tolerates one bad seed without allowing a split decision.
    - Alternatives: 5 of 5 (one outlier fails a cell); the median over seeds.
15. **D15. G2.**
    - Recommendation: prediction 1 holds in at least 3 of the 4 zero-bias cells.
    - Why: plan.md's G2 question is prediction 1, and 3 of 4 tolerates one architecture
      × depth exception.
    - Alternatives: all 4 cells; clause (b) alone; also requiring H1.
16. **D16. Hover check.**
    - Recommendation:
      - continuous LQR on the y-map's (A, B), Bryson Q and R, and a per-matrix sign
        threshold of 1e-3 × max|truth|;
      - H1 pre-registered as written;
      - 200 fixed trims from the no-drag trim family.
    - Why:
      - The y-map is the map the surrogate learns.
      - Bryson's rule makes Q and R unit-free.
      - Fixed trims cannot be chosen after the fact.
    - Alternatives:
      - discrete LQR on the RK4 map;
      - H1 reported only;
      - trims chosen by lens distance after training (rejected: chosen after seeing the
        model).
    - The trim code is not written yet. It follows this decision, before the tag.
17. **D17. Training.**
    - Recommendation: as in Training above.
    - Why: minibatch Adam, batch 2,048 and the evaluation interval are the author's
      decision of 4 Oct. lr 3e-3 and the GELU branch of width H follow lens/core.py.
    - Alternatives:
      - an lr sweep (it would need its own pre-registered selection rule);
      - evaluating every 1,000 steps (coarser early stopping).
18. **D18. Stopping rules.**
    - Recommendation: the analysis stops on any Corollary 1 deviation above 1e-10 (a
      code error) and on any manifest mismatch. `gather` refuses on any missing run,
      any SHA-256 mismatch, more than one commit, or a dirty tree.
    - Why: R6's rules, extended to array runs.
    - Alternatives: none proposed.
