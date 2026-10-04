# DRAFT: Pre-registration P-I, quadrotor surrogates

**Status: draft for the author and advisor. This is not a pre-registration file.**
When agreed, it becomes `prereg/p1.md`. The author tags it `prereg-p1`, and only then
may any P-I stage touch quadrotor data. Drafted 4 October 2026 from `docs/plan.md`,
`docs/theory.md`, `docs/STATE.md` and the infrastructure on branch `quadrotor-sim`.

> **Source caveat.** The repository holds no "protocol amendments" and no numbered
> predictions for P-I: `docs/plan.md` says only that they are to be written in
> `prereg/p1.md`. Predictions 1–4 below are therefore reconstructed from plan.md's
> P-I, hover-check and G2 text. Please replace them with the plan document's wording
> where it differs (decision D1). Prediction 5 is new, as requested.

## Question

Gate G2 (18 Oct): do zero-bias quadrotor surrogates concentrate Jacobian error at
hover, near the first-layer lens, and does that damage the hover linearisation? And
how does the lens move during training?

## Plant and data

- **Plant:** `plants/quadrotor.py` at the tagged commit.
  - 12 states: position, velocity, ZYX Euler angles, body rates.
  - 4 inputs: rotor thrusts in [0, f_max].
  - The author-approved Crazyflie 2.x parameters from gym-pybullet-drones' `cf2x.urdf`
    (889ce4a5c068ae4d811df1442ceb4f4d6cdf43eb), with g = 9.81 and no drag.
  - RK4 at dt = 0.01 s with zero-order hold.
- **Inputs to the surrogate:** z = (x, u) ∈ R¹⁶ (k = 16), z-scored per coordinate
  with the training split's mean and standard deviation.
- **Target:** y = (x_{t+1} − x_t)/dt ∈ R¹², z-scored the same way (plan.md).
  Main-text Jacobian errors are for this map; discrete-map values go to the appendix.
- **Ground truth:** J = ∂y/∂z by forward-mode autodiff of the simulator, in
  standardised coordinates J_std = diag(1/σ_y) J diag(σ_z).
- **Sampling** (infrastructure defaults; all open, see D3):
  - i.i.d. uniform in a box around hover: position ±1 m, velocity ±1 m/s, roll and
    pitch ±30°, yaw ±180°, body rates ±2 rad/s;
  - thrust u_i = u0 (1 ± 0.5) with u0 = m g/4, clipped to [0, f_max];
  - 20,000 train, 5,000 validation and 5,000 test states, from
    `numpy.random.default_rng(0)` drawn in that order.
  - Hover is the box centre, so in standardised coordinates it is near the origin,
    which is where a zero-bias lens starts (theory.md, Remark 1).
- **Storage:** the data stays outside the repository (`data/p1/`). Its SHA-256
  manifest (`results/p1/data_manifest.csv`) is committed, and every stage verifies it.

## Surrogates (the 40-model grid of plan.md)

- **Architectures:**
  - (a) Pre-norm residual block: h = E z + b; h ← h + W2 GELU(W1 (γ LN(h) + β) + b1)
    + b2; y = Wo h + bo.
  - (b) TD-MPC2-style NormedLinear stack: Linear → LayerNorm (affine) → Mish, with a
    Linear head.
- **Initialisations:**
  - `zero_bias`: PyTorch-default weights U(±1/√fan_in), every bias 0.
  - `torch_default`: weights and biases U(±1/√fan_in).
  - The two arms share weights within a seed.
- **Grid:** H = 128; 1 or 3 blocks; ε = 1e-5; 5 seeds. That is 2 × 2 × 2 × 5 = 40
  models.
- **Training:** float64; full-batch Adam, lr 3e-3, (0.9, 0.999, 1e-8); budget 100,000
  steps.
  - Early stopping: patience 10% of steps, tolerance 1% on held-out one-step MSE,
    evaluated every 500 steps.
  - The returned parameters are those with the lowest held-out MSE.
- **Lens log** (`lens/geometry.py`), every 500 steps: z*, κ, ‖c⊥‖, principal widths,
  and r*(d), r_eff(d), D(d)/r_eff(d) and S(d) along d₁.
- **The lens** is the first LayerNorm's (theory.md, Theorem 1). Distances are ρ, or
  ρ_eff = ‖A(z − z*)‖²/(‖c⊥‖² + Hε) for a degenerate lens.

Values computed before any quadrotor data exists, from initialisation alone, at
H = 128, k = 16, ε = 1e-5. Inputs are standardised i.i.d. uniform, 200 draws, d a
random unit direction:

| Initialisation | r_eff(d)/D, p5 / p50 / p95 | κ | ‖z*‖ |
| --- | --- | --- | --- |
| zero_bias | 0.0104 / 0.0113 / 0.0126 | ∞ (degenerate) | 0 |
| torch_default | 0.43 / 0.48 / 0.54 | median 5.6e-4 | median 0.38 |

## Predictions

Each prediction is evaluated per (architecture × depth) cell: 4 cells per
initialisation, 5 seeds each. "Holds in a cell" means it holds in at least 4 of the 5
seeds (D9).

1. **Error concentration near the lens** (G2's core; reconstructed).
   - Statistic: R = median relative Frobenius Jacobian error of the 10% of test states
     nearest the lens centre, divided by that of the 50% farthest (by ρ, or ρ_eff if
     degenerate).
   - Prediction: for zero-bias surrogates, R ≥ 2 in every cell.
2. **Default dependence** (reconstructed; R6's headline).
   - Prediction: in every cell, the zero-bias arm's median R over seeds exceeds the
     PyTorch-default arm's.
   - The comparison is paired by seed (shared weights) and made at held-out rel-MSE
     within 20% between the arms (plan.md P-II's matching rule; D10).
3. **Hover linearisation damage** (reconstructed; plan.md's hover check).
   - Prediction: at hover, zero-bias surrogates have a larger relative Frobenius error
     of (A, B) than PyTorch-default ones, paired by seed, in every cell.
   - Reported regardless:
     - sign agreement (entries above 1e-3 × max|truth|);
     - the LQR gain from the surrogate (Bryson Q, R from the sampling half-widths)
       against the true gain;
     - the closed-loop spectral abscissa of the surrogate's gain on the true
       linearisation, and stable or not.
4. **Mechanism** (reconstructed).
   - Prediction: on every trained model whose lens is non-degenerate, the first-layer
     LayerNorm output matches Corollary 1 to 1e-10 relative along D7-style lines: through
     μ along d₁, and through 10 data states along d₁.
   - For the 3-block models, attenuation (output peak/median relative to block 1's
     along z* + t·d) is reported regardless.
   - A failure of this check would indicate a code error, not a scientific outcome:
     the stage stops (as in R6's D7 (4)).
5. **Prospective lens migration and widening** (new).
   - Prediction: during training, the zero-bias lens leaves its degenerate start at
     the data centre, and its width moves toward the order of the data's half-width.
     Concretely, at the returned (best) parameters, in every zero-bias cell, all
     three of these hold:
     - (a) **leaves the degenerate start:** the lens is non-degenerate with κ ≤ 0.1,
       i.e. ‖c⊥‖² ≥ 10 Hε, so the offset c⊥, not ε, sets the width.
     - (b) **widens:** r_eff(d₁)/D(d₁) ≥ 0.1. That is about 9 times the initial median
       (0.0113) and 8 times its 95th percentile (0.0126).
     - (c) **toward order 1, not beyond:** r_eff(d₁)/D(d₁) ≤ 10.
   - Reported regardless: the full trajectories of κ, ‖z*‖, w(z*, μ) and
     r_eff(d)/D(d) over the log steps, for both initialisations.
   - Why these thresholds:
     - R6's exploratory E3 found TD-MPC2's trained first-layer lenses at
       r_eff(d₁)/D of 0.25–1.17. Their initialisation (trunc-normal, zero bias) put
       the start at 1e-4–0.085.
     - κ fell from ∞ to at most 0.0017 (lens_summary.csv).
     - 0.1 is an order of magnitude above the zero-bias start, and a factor of 2.5
       below the smallest trained TD-MPC2 value, so passing it is not guaranteed by
       random motion.
     - 10 caps "order of 1" symmetrically on a log scale.
     - κ ≤ 0.1 is lenient against TD-MPC2's 0.0017 but separates "width set by c⊥"
       from "width set by ε".
   - Caveat (D6): d₁ is poorly determined for standardised i.i.d. inputs, whose
     covariance is close to I. An alternative evaluates (b)–(c) along u_min, the
     lens's narrowest principal direction, or as the median over principal directions.

## Gate G2

Proposed (D11): G2 passes if prediction 1 holds in at least 3 of the 4 zero-bias cells.

## Reported regardless of outcome

For every model:
- the training history and lens log;
- the final lens (z*, κ, widths, w(z*, μ), coverage and sharpness along d₁ and
  u_min);
- R and both set statistics;
- the Jacobian error distribution;
- the Corollary 1 deviations and attenuation;
- the full hover check;
- for every stage: the config, the commit, the package versions and the data
  manifest.

## Decisions for the author

1. **D1. Predictions 1–4:** replace the reconstructed wording with the plan
   document's (and the "protocol amendments" mentioned in the queue, which are not in
   the repository).
2. **D2. Plant:** confirm g = 9.81, no drag (no linear drag coefficient is approved),
   RK4 at dt = 0.01 s, and position states included as inputs (k = 16).
3. **D3. Sampling design:**
   - i.i.d. box against trajectories;
   - the ranges (±1 m, ±1 m/s, ±30°, yaw ±180°, ±2 rad/s, thrust ±50%);
   - uniform against Gaussian;
   - the sizes 20,000 / 5,000 / 5,000;
   - the data seed.
4. **D4. Standardisation:** a per-coordinate z-score from the training split, or a
   fixed physical scaling.
5. **D5. Jacobian error metric:** relative Frobenius per state in standardised
   coordinates. Alternatives: absolute; physical units; separate A and B blocks.
6. **D6. Direction for sharpness and prediction 5:** d₁ (r6.md's definition, poorly
   determined here), u_min, or the median over principal directions. Also: D as
   (q97.5 − q2.5)/2.
7. **D7. Near/far split:** 10% against 50%, the median, sets of ceil(fraction · n);
   ρ against ρ_eff.
8. **D8. Thresholds:**
   - R ≥ 2 (prediction 1);
   - κ ≤ 0.1 and 0.1 ≤ r_eff/D ≤ 10 (prediction 5);
   - Corollary 1 tolerance 1e-10;
   - the sign threshold of 1e-3 × max|truth|.
9. **D9. Seeds rule:** "holds in a cell" = at least 4 of 5 seeds.
10. **D10. Matched accuracy** for predictions 2–3: rel-MSE within 20% (plan.md P-II),
    or no matching.
11. **D11. The G2 rule:** at least 3 of 4 zero-bias cells for prediction 1. Does G2
    also need prediction 3?
12. **D12. Hover check details:**
    - continuous LQR on the y-map's (A, B);
    - Bryson Q, R from the half-widths;
    - "steady-flight trims spanning lens distance" (plan.md) — which trims?
13. **D13. Training:**
    - lr 3e-3; full batch;
    - budget 100k;
    - evaluation and logging every 500 steps;
    - restore of the best parameters;
    - branch width = H and GELU for the pre-norm arm.
14. **D14. Prediction 5's scope:** zero-bias only, or also a prediction for the
    PyTorch-default arm (it starts at r_eff/D ≈ 0.48, already of order 1)?
15. **D15. Stopping rules:** the stage stops on a Corollary 1 deviation above 1e-10
    and on any manifest mismatch (as in R6).
